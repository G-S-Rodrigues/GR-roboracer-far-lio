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
"""FARLIO_1870: a told start pose reaches the EKF through its first lidar measurement, for both 2D filters.

tam_gicp_ext with wait_initial_pose on a prior .pcd of the oval written under T_START = (2.0, -1.0, 30 deg), plus
tam_state_estimation_node_ekf_2d_kin or _nh (case kin64 or nh64, env FARLIO_LAP_CASE as in FARLIO_1610; the nh case
also gets per-wheel speeds). The test streams 3 s of standing scans and IMU into the hold, publishes the pose
(T_START moved by 0.1 m / 0.05 m / 2 deg, inside the matcher's acceptance ellipsis; see FARLIO_1820), then drives 5 s of FARLIO_1610's ramp profile (2 s ramp to 2 m/s, then
cruise) from T_START. Asserts: the first /core/state/odometry is within BOUNDS_1870_START of T_START; TF
far_lio_odom -> base_link at that stamp agrees with it; the error 5 s after the start is within BOUNDS_1870_END.
"""
import math
import os
import sys
import tempfile
import unittest

from launch import LaunchDescription
from launch_ros.actions import Node
import launch_testing
import launch_testing.actions
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # launch_test does not put the test directory on the path
import farlio_synthetic as fs  # noqa: E402
from farlio_stream import compose, pose_error, RATE, Rig  # noqa: E402

CASE = os.environ.get('FARLIO_LAP_CASE', 'kin64')
EKF_EXEC = 'tam_state_estimation_node_ekf_' + ('2d_nh' if CASE.startswith('nh') else '2d_kin')
CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'config', 'f1tenth.yml')
T_START = (2.0, -1.0, math.radians(30.0))
OFFSET = (0.1, 0.05, math.radians(2.0))   # inside the matcher's acceptance ellipsis (0.15 x 0.075 m): see the docstring
MAP_PCD = os.path.join(tempfile.mkdtemp(prefix='farlio_1870_'), 'map.pcd')
SPEED, RAMP_S, DRIVE_S, HOLD_S = 2.0, 2.0, 5.0, 3.0
# Measured 2026-11-02, 10 runs per case (load < 8), offset 0.1 / 0.05 m / 2 deg. First EKF pose from T_START: 0.0480 m /
# 0.190 deg in all 20 runs, so the tight target 5 cm / 1 deg is asserted. TF far_lio_odom -> base_link against the EKF
# odometry at its stamp: 0.00000 m / 0.0000 deg in all 20, target 1 mm / 0.01 deg. Error 5.27 s after the start (GICP
# loses the along-track position on the straight, gotcha-18 style; this is a record, not an accuracy):
#   kin64  0.5300 0.5296 0.5118 0.5274 0.5271 0.5295 0.5280 0.5100 0.5317 0.5277 m   (worst 0.5317 -> 1.07)
#          0.513 0.519 1.008 0.537 0.540 0.522 0.150 1.028 0.499 0.533 deg           (worst 1.028 -> 2.06)
#   nh64   0.5090 0.5068 0.5103 0.5053 0.5086 0.5112 0.4875 0.4986 0.5076 0.5143 m   (worst 0.5143 -> 1.03)
#          0.380 0.386 0.442 0.442 0.416 0.390 0.498 0.433 0.368 0.380 deg           (worst 0.498 -> 1.00)
BOUNDS_1870_START = {'kin64': (0.05, 1.0), 'nh64': (0.05, 1.0)}
BOUNDS_1870_END = {'kin64': (1.07, 2.06), 'nh64': (1.03, 1.00)}
TF_AGREE = (0.001, 0.01)


@pytest.mark.launch_test
def generate_test_description():
    fs.write_map_pcd(MAP_PCD, T_START)
    tf = Node(package='tf2_ros', executable='static_transform_publisher',
              arguments=['--x', str(fs.LASER_X), '--z', str(fs.LASER_Z), '--frame-id', 'base_link',
                         '--child-frame-id', 'laser'])
    odometry = Node(package='tam_odometry', executable='tam_gicp_ext', namespace='/core/state',
                    name='LidarOdometry', output='screen',
                    parameters=[CONFIG, {'use_sim_time': True, 'pipeline.update_map': False,
                                         'node.input_map': MAP_PCD, 'node.wait_initial_pose': True}])
    ekf = Node(package='tam_state_estimation_node', executable=EKF_EXEC, namespace='/core/state',
               name='StateEstimation', output='screen', parameters=[CONFIG, {'use_sim_time': True}])
    return LaunchDescription([tf, odometry, ekf, launch_testing.actions.ReadyToTest()])


class TestPoseThroughEkf(unittest.TestCase):

    def setUp(self):
        self.rig = Rig(wheels=CASE.startswith('nh'))

    def tearDown(self):
        self.rig.close()

    def test_farlio_1870_ekf_starts_at_the_told_pose(self):
        rig = self.rig
        rig.wait_graph(imu_sub=True)
        rig.run_standing(HOLD_S)
        self.assertEqual(len(rig.ekf), 0, 'EKF odometry during the hold')
        rig.publish_pose(compose(T_START, OFFSET))
        dt, i0 = 1.0 / RATE, rig.i
        v = s = 0.0
        for _ in range(int(DRIVE_S * RATE)):
            a = SPEED / RAMP_S if v < SPEED - 1e-9 else 0.0
            v = min(SPEED, v + a * dt)
            s += v * dt
            x, y, yaw, k = fs.centerline_pose(s)
            rig.tick(rig.frame(x, y, yaw), v, a, v * k)
        rig.drain(0.3)
        self.assertGreater(len(rig.ekf), 0, 'no EKF odometry after the pose')
        first = rig.ekf[0]
        d, dyaw = pose_error(first[1:], T_START)
        self.assertLess(d, 1.0, 'first EKF pose %.2f m from the start' % d)
        # the EKF's last pose against the truth at its stamp
        last = rig.ekf[-1]
        k_last = min(max(int(round((last[0] - 1.0) / dt)) - i0, 0), int(DRIVE_S * RATE) - 1)
        v = s = 0.0
        for _ in range(k_last + 1):
            a = SPEED / RAMP_S if v < SPEED - 1e-9 else 0.0
            v = min(SPEED, v + a * dt)
            s += v * dt
        x, y, yaw, _ = fs.centerline_pose(s)
        end_d, end_dyaw = pose_error(last[1:], compose(T_START, (x, y, yaw)))
        # TF at the first EKF stamp
        tf = min(rig.tf, key=lambda q: abs(q[0] - first[0]))
        tf_d, tf_dyaw = pose_error(tf[1:], first[1:])
        print('FARLIO_1870 %s first EKF pose %.4f m %.3f deg from T_START; TF %.5f m %.4f deg from it (dt %.4f s); '
              'end error %.4f m %.3f deg at t=%.2f s' % (CASE, d, dyaw, tf_d, tf_dyaw, abs(tf[0] - first[0]),
                                                         end_d, end_dyaw, last[0] - first[0]))
        bs, be = BOUNDS_1870_START[CASE], BOUNDS_1870_END[CASE]
        self.assertIsNotNone(bs, 'bounds measured')
        self.assertIsNotNone(be, 'bounds measured')
        self.assertIsNotNone(TF_AGREE, 'bounds measured')
        self.assertLessEqual(d, bs[0], 'first EKF pose %.4f m from T_START' % d)
        self.assertLessEqual(dyaw, bs[1], 'first EKF pose %.3f deg from T_START' % dyaw)
        self.assertLessEqual(tf_d, TF_AGREE[0], 'TF differs from the EKF odometry by %.5f m' % tf_d)
        self.assertLessEqual(tf_dyaw, TF_AGREE[1], 'TF differs from the EKF odometry by %.4f deg' % tf_dyaw)
        self.assertLessEqual(end_d, be[0], 'end error %.4f m' % end_d)
        self.assertLessEqual(end_dyaw, be[1], 'end error %.3f deg' % end_dyaw)
