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
"""FARLIO-1050: the 2D EKF with the RoboRacer gym's inputs, on unmodified upstream.

One IMU plus lidar odometry as the only position / orientation / velocity source, along a
2 m/s circle of radius 5 m. Variants (env FARLIO_VARIANT, one launch test each):

  zero_accel  ekf_2d_kin, /imu linear_acceleration = 0, exact yaw rate. Characterisation: it is
              EXPECTED to diverge. The 2D state velocity is in the body frame
              (ekf_2d.hpp, state transition: vx' = ax + w*vy, vy' = ay - w*vx), so the IMU must
              carry the centripetal term a_y = v*w; with a = 0 the model rotates the body-frame
              velocity and drifts.
  body_accel  ekf_2d_kin, /imu linear_acceleration in the body frame (a_x = dv/dt = 0,
              a_y = v*w = 0.4 m/s^2, a_z = +9.81), which is what the sim will publish.
  nh          ekf_2d_nh, body-frame accel, plus per-wheel speeds from the circle kinematics, plus synthetic /vehicle/sensor/wheelspeed_radps
              (tum_msgs/TUMFloat64PerWheelStamped) and /vehicle/sensor/wheelspeed_status.

It asserts /core/state/odometry >= 0.9 x 100 Hz and records the worst position error.
The measurement twist is in the body frame: state_estimation.hpp (set_input_linear_velocity,
2D branch) rotates it only by pitch and roll (vector_to_2d), never by yaw.
"""
import math
import os
import time
import unittest

from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from launch import LaunchDescription
from launch_ros.actions import Node
import launch_testing
import launch_testing.actions
from nav_msgs.msg import Odometry
import pytest
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Imu
from tum_msgs.msg import TUMFloat64PerWheelStamped

PARAMS = os.path.join(os.path.dirname(__file__), 'ekf_2d_gym_like.yml')
RADIUS = 5.0
SPEED = 2.0
RATE = 100.0
WHEEL_R = 0.058       # f1tenth.urdf wheel radius
HALF_TRACK = 0.118    # f1tenth.urdf half track
LR = 0.17145          # base_link to rear axle
LF = 0.33 - LR        # wheelbase 0.33 m
VARIANT = os.environ.get('FARLIO_VARIANT', 'zero_accel')
MIN_ERR_M = float(os.environ.get('FARLIO_MIN_ERR_M', '0'))
MAX_STAMP_AGE_S = float(os.environ.get('FARLIO_MAX_STAMP_AGE_S', '1.0'))
MAX_ERR_M = float(os.environ.get('FARLIO_MAX_ERR_M', '0.05'))


@pytest.mark.launch_test
def generate_test_description():
    ekf = Node(
        package='tam_state_estimation_node',
        executable='tam_state_estimation_node_ekf_2d_nh' if VARIANT == 'nh' else 'tam_state_estimation_node_ekf_2d_kin',
        parameters=[PARAMS],
        output='screen')
    return LaunchDescription([ekf, launch_testing.actions.ReadyToTest()])


class TestEkfGymLike(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = rclpy.create_node('farlio_1050_driver')

    @classmethod
    def tearDownClass(cls):
        cls.node.destroy_node()
        rclpy.shutdown()

    def test_runs_and_tracks(self):
        """FARLIO-1050."""
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        pub_imu = self.node.create_publisher(Imu, '/imu', qos)
        pub_odom = self.node.create_publisher(Odometry, '/core/state/lidar_odometry', qos)
        pub_status = self.node.create_publisher(
            DiagnosticArray, '/core/state/lidar_odometry_status', qos)
        received = []
        self.node.create_subscription(
            Odometry, '/core/state/odometry', lambda m: received.append(
                (time.monotonic(), m, self.node.get_clock().now().nanoseconds)), qos)

        omega = SPEED / RADIUS
        pub_wheel = self.node.create_publisher(TUMFloat64PerWheelStamped,
                                               '/vehicle/sensor/wheelspeed_radps', qos)
        pub_wst = self.node.create_publisher(DiagnosticArray,
                                             '/vehicle/sensor/wheelspeed_status', qos)
        t0 = time.monotonic()
        next_tick = t0
        end = t0 + 8.0
        truth = {}
        while time.monotonic() < end:
            now = time.monotonic()
            if now >= next_tick:
                next_tick += 1.0 / RATE
                t = now - t0
                stamp = self.node.get_clock().now().to_msg()
                th = omega * t
                imu = Imu()
                imu.header.stamp = stamp
                imu.header.frame_id = 'base_link'
                imu.angular_velocity.z = omega
                if VARIANT in ('body_accel', 'nh'):
                    imu.linear_acceleration.y = SPEED * omega
                    imu.linear_acceleration.z = 9.81
                pub_imu.publish(imu)
                odom = Odometry()
                odom.header.stamp = stamp
                odom.header.frame_id = 'local_cartesian'
                odom.child_frame_id = 'base_link'
                odom.pose.pose.position.x = RADIUS * math.sin(th)
                odom.pose.pose.position.y = RADIUS * (1.0 - math.cos(th))
                odom.pose.pose.orientation.z = math.sin(th / 2.0)
                odom.pose.pose.orientation.w = math.cos(th / 2.0)
                # body-frame twist (ROS child frame): see the module docstring
                odom.twist.twist.linear.x = SPEED
                odom.twist.twist.angular.z = omega
                for i in range(6):
                    odom.pose.covariance[7 * i] = 0.01
                    odom.twist.covariance[7 * i] = 0.01
                pub_odom.publish(odom)
                st = DiagnosticArray()
                st.header.stamp = stamp
                st.status.append(DiagnosticStatus(level=DiagnosticStatus.OK, name='lidar'))
                pub_status.publish(st)
                if VARIANT == 'nh':
                    w = TUMFloat64PerWheelStamped()
                    w.stamp = stamp
                    # Per-wheel speeds from the circle's kinematics (left turn, ICR at body y = +R):
                    # each wheel turns at omega * its distance to the ICR / wheel radius.
                    def wheel(x, y):
                        return omega * math.hypot(x, y - RADIUS) / WHEEL_R
                    w.data.front_left = wheel(LF, HALF_TRACK)
                    w.data.front_right = wheel(LF, -HALF_TRACK)
                    w.data.rear_left = wheel(-LR, HALF_TRACK)
                    w.data.rear_right = wheel(-LR, -HALF_TRACK)
                    pub_wheel.publish(w)
                    pub_wst.publish(st)
                truth[(stamp.sec, stamp.nanosec)] = (odom.pose.pose.position.x,
                                                     odom.pose.pose.position.y)
            rclpy.spin_once(self.node, timeout_sec=0.0005)

        # the first seconds are initialization; judge the last 4 s
        window = [m for (tm, m, _) in received if tm > t0 + 4.0]
        recv_ns = [rn for (tm, _, rn) in received if tm > t0 + 4.0]
        self.assertGreater(len(window), 0, 'the EKF published nothing: it never left init/ERROR')
        span = [tm for (tm, m, _) in received if tm > t0 + 4.0]
        rate = (len(span) - 1) / (span[-1] - span[0])
        self.assertGreaterEqual(rate, 0.9 * RATE, f'odometry rate {rate:.1f} Hz')
        worst = 0.0
        # Output-stamp lag: how old the stamp is when the message arrives. Stamps must not be
        # in the future; the age bound is MAX_STAMP_AGE_S.
        ages = [(rn - (m.header.stamp.sec * 10**9 + m.header.stamp.nanosec)) / 1e9
                for m, rn in zip(window, recv_ns)]
        print(f'FARLIO_1050 {VARIANT} stamp age min {min(ages):.4f} max {max(ages):.4f} s')
        self.assertGreaterEqual(min(ages), -0.001, 'output stamp is in the future')
        self.assertLessEqual(max(ages), MAX_STAMP_AGE_S, 'output stamp lags too far')
        # Output stamps lag the input by average_delay_ms: compare to the nearest input stamp
        # position error against the closest published truth sample
        keys = sorted(truth)
        for m in window:
            k = (m.header.stamp.sec, m.header.stamp.nanosec)
            best = min(keys, key=lambda c: abs((c[0] - k[0]) * 1e9 + (c[1] - k[1])))
            x, y = truth[best]
            err = math.hypot(m.pose.pose.position.x - x, m.pose.pose.position.y - y)
            worst = max(worst, err)
        print(f'FARLIO_1050 {VARIANT} worst position error {worst:.4f} m')
        if MIN_ERR_M > 0.0:
            # characterisation variant: the documented failure must still happen
            self.assertGreater(worst, MIN_ERR_M, f'expected divergence, worst {worst:.3f} m')
        else:
            self.assertLessEqual(worst, MAX_ERR_M, f'worst position error {worst:.3f} m')
