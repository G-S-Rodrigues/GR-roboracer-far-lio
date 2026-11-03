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
"""Rig for the part 4b launch tests (FARLIO_1820..1880): a driver node that ticks a 100 Hz sim clock with /imu (and
wheel speeds), streams 64-beam scans cast from farlio_synthetic, publishes a (latched) initial pose, and records the
lidar odometry, the EKF odometry and TF far_lio_odom -> base_link. Not a test file."""
import math
import time

import numpy as np
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
import rclpy
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Imu, LaserScan
from tf2_msgs.msg import TFMessage
from tum_msgs.msg import TUMFloat64PerWheelStamped

import farlio_synthetic as fs

RATE = 100.0
GRAVITY = 9.81
SEED = 20261031
INITIAL_POSE_TOPIC = '/core/state/initial_pose'


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def compose(base, rel):
    """base (x, y, yaw) composed with rel (x, y, yaw) given in base's frame."""
    c, s = math.cos(base[2]), math.sin(base[2])
    return (base[0] + c * rel[0] - s * rel[1], base[1] + s * rel[0] + c * rel[1], base[2] + rel[2])


def pose_error(a, b):
    """(position error m, absolute yaw error deg) between two (x, y, yaw_rad) poses."""
    dyaw = (a[2] - b[2] + math.pi) % (2.0 * math.pi) - math.pi
    return math.hypot(a[0] - b[0], a[1] - b[1]), abs(math.degrees(dyaw))


class Rig:

    def __init__(self, wheels=False, pace=1.0):
        rclpy.init()
        self.node = rclpy.create_node('farlio_18xx_driver')
        be = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        self.pub_clock = self.node.create_publisher(Clock, '/clock', QoSProfile(depth=10))
        self.pub_scan = self.node.create_publisher(LaserScan, '/scan', be)
        self.pub_imu = self.node.create_publisher(Imu, '/imu', be)
        self.pub_pose = self.node.create_publisher(PoseWithCovarianceStamped, INITIAL_POSE_TOPIC, latched)
        self.wheels = wheels
        if wheels:
            self.pub_wheel = self.node.create_publisher(TUMFloat64PerWheelStamped,
                                                        '/vehicle/sensor/wheelspeed_radps', be)
            self.pub_wst = self.node.create_publisher(DiagnosticArray, '/vehicle/sensor/wheelspeed_status', be)
        self.lidar, self.ekf, self.tf = [], [], []   # (stamp s, x, y, yaw rad)
        self.node.create_subscription(Odometry, '/core/state/lidar_odometry', lambda m: self.lidar.append(self._p(m)), be)
        self.lidar_status = []   # (stamp s, level, message) of /core/state/lidar_odometry_status
        self.node.create_subscription(DiagnosticArray, '/core/state/lidar_odometry_status', self._on_lstat, be)
        self.node.create_subscription(Odometry, '/core/state/odometry', lambda m: self.ekf.append(self._p(m)), be)
        self.node.create_subscription(TFMessage, '/tf', self._on_tf, QoSProfile(depth=100))
        self.dt = 1.0 / RATE
        self.pace = pace
        self.i = 0
        self.t_wall0 = time.monotonic()
        self.segs = fs._segments(0.05)
        self.rng = np.random.default_rng(SEED)

    @staticmethod
    def _p(m):
        p = m.pose.pose
        return (m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, p.position.x, p.position.y, yaw_of(p.orientation))

    def _on_lstat(self, msg):
        if msg.status:
            lv = msg.status[0].level
            self.lidar_status.append((msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                                      ord(lv) if isinstance(lv, bytes) else int(lv), msg.status[0].message))

    def status_summary(self):
        seen = []
        for _, lv, m in self.lidar_status:
            if not seen or seen[-1] != (lv, m):
                seen.append((lv, m))
        return seen[:6]

    def _on_tf(self, msg):
        for t in msg.transforms:
            if t.header.frame_id == 'far_lio_odom' and t.child_frame_id == 'base_link':
                tr = t.transform
                self.tf.append((t.header.stamp.sec + t.header.stamp.nanosec * 1e-9, tr.translation.x,
                                tr.translation.y, yaw_of(tr.rotation)))

    def close(self):
        self.node.destroy_node()
        rclpy.shutdown()

    def sim_time(self):
        return 1.0 + self.i * self.dt

    def frame(self, x, y, yaw):
        """One noisy scan (list of ranges) cast from the oval-frame pose (x, y, yaw)."""
        return [float(r) for r in fs.cast_scan(self.segs, x, y, yaw, self.rng)]

    def tick(self, scan=None, v=0.0, a=0.0, omega=0.0):
        """Publish one 100 Hz step of /clock and /imu (and wheel speeds, and a scan when given); pace to wall time."""
        ns = int(round(self.i * self.dt * 1e9)) + 1_000_000_000
        clock = Clock()
        clock.clock.sec, clock.clock.nanosec = divmod(ns, 10 ** 9)
        self.pub_clock.publish(clock)
        imu = Imu()
        imu.header.stamp = clock.clock
        imu.header.frame_id = 'base_link'
        imu.angular_velocity.z = omega
        imu.linear_acceleration.x = a                 # body frame: a_x = dv/dt, a_y = v * w
        imu.linear_acceleration.y = v * omega
        imu.linear_acceleration.z = GRAVITY
        self.pub_imu.publish(imu)
        if self.wheels:
            w = TUMFloat64PerWheelStamped()
            w.stamp = clock.clock
            w.data.front_left = fs._wheel(fs.LF, fs.HALF_TRACK, v, omega)
            w.data.front_right = fs._wheel(fs.LF, -fs.HALF_TRACK, v, omega)
            w.data.rear_left = fs._wheel(-fs.LR, fs.HALF_TRACK, v, omega)
            w.data.rear_right = fs._wheel(-fs.LR, -fs.HALF_TRACK, v, omega)
            self.pub_wheel.publish(w)
            st = DiagnosticArray()
            st.header.stamp = clock.clock
            st.status.append(DiagnosticStatus(level=DiagnosticStatus.OK, name='wheel'))
            self.pub_wst.publish(st)
        if scan is not None:
            s = LaserScan()
            s.header.stamp = clock.clock
            s.header.frame_id = 'laser'
            s.angle_min = -fs.FOV / 2.0
            s.angle_increment = fs.FOV / (fs.NUM_BEAMS - 1)
            s.angle_max = fs.FOV / 2.0
            s.range_min, s.range_max = fs.RANGE_MIN, fs.RANGE_MAX
            s.ranges = scan
            self.pub_scan.publish(s)
        self.i += 1
        while time.monotonic() - self.t_wall0 < self.i * self.dt * self.pace:
            rclpy.spin_once(self.node, timeout_sec=0.0005)

    def run_standing(self, seconds, scans=True, scan_pose=(0.0, 0.0, 0.0)):
        """Tick `seconds` of sim time standing still at `scan_pose` (oval frame), with or without scans."""
        for _ in range(int(seconds * RATE)):
            self.tick(self.frame(*scan_pose) if scans else None)

    def wait_graph(self, timeout_s=20.0, pose_sub=True, imu_sub=False):
        """Tick (clock + IMU only) until a /scan subscriber exists (and an /imu one, when asked) and, when asked, the
        initial-pose topic has one.

        Returns (scan_sub_seen, pose_sub_seen). The pose subscriber may never appear (before it is implemented, or
        when the node does not take the input), so that wait is bounded and the caller carries on.
        """
        end = time.monotonic() + timeout_s
        while time.monotonic() < end and (self.pub_scan.get_subscription_count() < 1 or
                                          (imu_sub and self.pub_imu.get_subscription_count() < 1)):
            self.tick()
        scan_ok = self.pub_scan.get_subscription_count() >= 1
        end = time.monotonic() + (timeout_s / 2.0 if pose_sub else 0.0)
        while time.monotonic() < end and self.pub_pose.get_subscription_count() < 1:
            self.tick()
        return scan_ok, self.pub_pose.get_subscription_count() >= 1

    def publish_pose(self, pose, frame='far_lio_odom', zero_quaternion=False):
        m = PoseWithCovarianceStamped()
        m.header.frame_id = frame
        m.header.stamp.sec, m.header.stamp.nanosec = divmod(int(round(self.sim_time() * 1e9)), 10 ** 9)
        m.pose.pose.position.x, m.pose.pose.position.y = pose[0], pose[1]
        m.pose.pose.orientation.z, m.pose.pose.orientation.w = math.sin(pose[2] / 2.0), math.cos(pose[2] / 2.0)
        if zero_quaternion:
            m.pose.pose.orientation.z = m.pose.pose.orientation.w = 0.0
        m.pose.covariance[0] = m.pose.covariance[7] = 0.25
        m.pose.covariance[35] = 0.07
        self.pub_pose.publish(m)

    def drain(self, seconds=0.5):
        for _ in range(int(seconds * RATE)):
            self.tick()
