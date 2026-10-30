# Copyright 2026 Guilherme Rodrigues
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""FARLIO-1410..1440: the LaserScan input of tam_icp_ext (launch_testing).

The node runs with node.input_scan:=/scan, node.wait_tf:=false and pipeline.lidar_2d:=true next to a static
TF base_link -> laser (0.10355, 0, 0.031). The registered frame is read from /core/state/frame_registration
(debug mode), which is published best-effort: a QoS mismatch looks like absent data, not like an error.
"""
import math
import time
import unittest

import launch
import launch_ros.actions
import launch_testing.actions
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan, PointCloud2
from sensor_msgs_py import point_cloud2

LASER_X, LASER_Z = 0.10355, 0.031
INF, NAN = float("inf"), float("nan")
BEST_EFFORT = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)


def _node_params(extra):
    params = {
        "node.wait_tf": False,
        "node.odom_frame": "odom",
        "node.child_frame": "base_link",
        "node.cloud_frame": "laser",
        "pipeline.lidar_2d": True,
        "pipeline.update_map": True,
        "pipeline.debug_mode": True,
        "pipeline.undistort": False,
        "pipeline.preprocess": False,
        "pipeline.downsample": False,
        "pipeline.buffer_z": False,
        "diagnostic.check_input_status": False,
        "map.voxel_size": 0.5,
    }
    params.update(extra)
    return params


def generate_test_description():
    tf = launch_ros.actions.Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=["--x", str(LASER_X), "--z", str(LASER_Z), "--frame-id", "base_link", "--child-frame-id", "laser"],
    )
    scan_node = launch_ros.actions.Node(
        package="tam_odometry",
        executable="tam_icp_ext",
        name="scan_odometry",
        parameters=[_node_params({"node.input_scan": "/scan"})],
        output="screen",
    )
    # the default input: PointCloud2 on node.input_pointcloud (FARLIO-1440)
    cloud_node = launch_ros.actions.Node(
        package="tam_odometry",
        executable="tam_icp_ext",
        name="cloud_odometry",
        parameters=[_node_params({"node.input_pointcloud": "/cloud_in"})],
        remappings=[("/core/state/frame_registration", "/cloud_node/frame_registration")],
        output="screen",
    )
    return launch.LaunchDescription([tf, scan_node, cloud_node, launch_testing.actions.ReadyToTest()])


def _room_scan(n=90, r=3.0, range_min=0.1, range_max=10.0):
    """A scan around the full circle at about 3 m. 90 beams: neighbours are 0.2 m apart, wider than the registration's
    voxel/6.3 downsampling (0.5 m / 6.3 = 0.08 m), so no two beams share a cell and the registered frame keeps every point."""
    scan = LaserScan()
    scan.header.frame_id = "laser"
    scan.angle_min = -math.pi
    scan.angle_increment = 2.0 * math.pi / n
    scan.angle_max = scan.angle_min + (n - 1) * scan.angle_increment
    scan.range_min, scan.range_max = range_min, range_max
    scan.ranges = [r + 0.5 * math.sin(3.0 * i * scan.angle_increment) for i in range(n)]
    return scan


class TestScanInput(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = rclpy.create_node("scan_input_test")

    @classmethod
    def tearDownClass(cls):
        cls.node.destroy_node()
        rclpy.shutdown()

    def _frame_for(self, scan, pub_topic="/scan", sub_topic="/core/state/frame_registration", timeout=40.0):
        """Publish `scan` best-effort at 20 Hz until a registered frame arrives; None on timeout."""
        pub = self.node.create_publisher(LaserScan, pub_topic, BEST_EFFORT)
        got = []
        sub = self.node.create_subscription(PointCloud2, sub_topic, got.append, BEST_EFFORT)
        end = time.time() + timeout
        while time.time() < end and not got:
            scan.header.stamp = self.node.get_clock().now().to_msg()
            pub.publish(scan)
            rclpy.spin_once(self.node, timeout_sec=0.05)
        self.node.destroy_subscription(sub)
        self.node.destroy_publisher(pub)
        return got[0] if got else None

    @staticmethod
    def _points(cloud):
        return [(p[0], p[1], p[2]) for p in point_cloud2.read_points(cloud, field_names=("x", "y", "z"))]

    def test_farlio_1410_best_effort_scan_is_registered(self):
        frame = self._frame_for(_room_scan())
        self.assertIsNotNone(frame, "no /core/state/frame_registration: scan QoS or the LaserScan subscription is wrong")

    def test_farlio_1420_invalid_beams_are_dropped(self):
        scan = _room_scan()
        n_before = len(scan.ranges)
        scan.ranges[0], scan.ranges[1] = INF, NAN
        scan.ranges[2] = scan.range_min / 2.0
        scan.ranges[3] = scan.range_max * 1.2
        frame = self._frame_for(scan)
        self.assertIsNotNone(frame)
        self.assertEqual(len(self._points(frame)), n_before - 4)

    def test_farlio_1430_points_lie_at_the_declared_angles(self):
        scan = _room_scan()
        frame = self._frame_for(scan)
        self.assertIsNotNone(frame)
        got = self._points(frame)
        self.assertEqual(len(got), len(scan.ranges))
        # the registration may reorder points, so match each expected point to the nearest published one
        for i, r in enumerate(scan.ranges):
            a = scan.angle_min + i * scan.angle_increment
            want = (r * math.cos(a) + LASER_X, r * math.sin(a), LASER_Z)
            nearest = min(math.dist(want, p) for p in got)
            self.assertLess(nearest, 2e-3, "beam %d (angle %.4f rad) has no point near %s" % (i, a, want))

    def test_farlio_1440_default_input_is_the_pointcloud(self):
        """S2: with node.input_scan empty the node has a PointCloud2 subscription and no LaserScan one."""
        topics = dict(self.node.get_topic_names_and_types())
        self.assertIn("/scan", topics)  # the scan node subscribes it ...
        infos = self.node.get_subscriptions_info_by_topic("/scan")
        self.assertEqual({i.node_name for i in infos if i.topic_type == "sensor_msgs/msg/LaserScan"}, {"scan_odometry"})
        cloud_infos = self.node.get_subscriptions_info_by_topic("/cloud_in")
        self.assertEqual({i.node_name for i in cloud_infos}, {"cloud_odometry"})
        self.assertEqual({i.topic_type for i in cloud_infos}, {"sensor_msgs/msg/PointCloud2"})
        # ... and a published cloud produces a registered frame (the 3D path)
        pub = self.node.create_publisher(PointCloud2, "/cloud_in", BEST_EFFORT)
        got = []
        sub = self.node.create_subscription(PointCloud2, "/cloud_node/frame_registration", got.append, BEST_EFFORT)
        scan = _room_scan()
        pts = [
            (r * math.cos(scan.angle_min + i * scan.angle_increment), r * math.sin(scan.angle_min + i * scan.angle_increment), 0.0)
            for i, r in enumerate(scan.ranges)
        ]
        end = time.time() + 40.0
        while time.time() < end and not got:
            header = scan.header
            header.stamp = self.node.get_clock().now().to_msg()
            pub.publish(point_cloud2.create_cloud_xyz32(header, pts))
            rclpy.spin_once(self.node, timeout_sec=0.05)
        self.assertTrue(got, "the PointCloud2 node produced no registered frame")
