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
"""FARLIO_1820..1880: the initial-pose input of the scan matcher (part 4b), on the synthetic oval.

The told pose is T_START moved by OFFSET = (0.1 m, 0.05 m, 2 deg), and no more: the matcher's acceptance check
(`ellipsis_check`, ellipsis_size_s 0.15 m x ellipsis_size_d 0.075 m around the guess, config/f1tenth.yml:135-136)
marks a registration further than that from its guess "converged | too far from guess" (level 2), and the EKF then
never initialises. Measured 2026-11-02: offsets 0/0/0, 0.05/0.02/1 deg and 0.1/0.05/2 deg start the EKF; the original
0.3/0.2/5 deg does not (FARLIO_1820 failed: no EKF odometry within 2 s). The matcher also barely corrects an offset
(first odometry 0.051 m from T_START at a 0.054 m offset). A told pose must already be accurate to about 0.1 m /
2 deg; correcting a rough pose is part 4c's.

tam_gicp_ext (2D lidar, LaserScan input; env FARLIO_IP_EXEC=tam_cuda_gicp_ext for the CUDA twin) localizes on a prior
.pcd of the oval written under T_START = (2.0, -1.0, 30 deg), so the car's start is not the map origin: a scan
registered at the identity is 2.24 m / 30 deg off. The case (env FARLIO_IP_CASE, set in CMakeLists.txt) chooses what
the test does with /core/state/initial_pose (geometry_msgs/PoseWithCovarianceStamped, RELIABLE TRANSIENT_LOCAL, frame
far_lio_odom). A case defines only its own test class, so no test is skipped:

    told              FARLIO_1830  a pose 0.1 m / 0.05 m / 2 deg off T_START is the first scan's guess; every lidar odometry of
                                   the following second lands within BOUNDS_1830 of T_START (FARLIO_1880 is this case
                                   on tam_cuda_gicp_ext)
    untold            FARLIO_1830  control: no pose, the first lidar odometry is >= 1 m away (the test bites)
    hold              FARLIO_1820  wait_initial_pose: 3 s of scans and IMU, no lidar odometry and no EKF odometry; the
                                   pose then releases both
    wrong_frame       FARLIO_1840  a pose in frame `map` is rejected with a WARN and the hold continues
    late_pose         FARLIO_1850  a pose after the first registered scan is ignored (WARN), no jump
    wait_tf_conflict  FARLIO_1860  wait_initial_pose with wait_tf: the node refuses to start
    icp_cv_refused    FARLIO_1860  a topic on a non-EXTERNALGUESS model (tam_icp_cv): the node refuses to start
"""
import math
import os
import re
import sys
import tempfile
import unittest

from launch import LaunchDescription
from launch_ros.actions import Node
import launch_testing
import launch_testing.actions
import launch_testing.asserts
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # launch_test does not put the test directory on the path
import farlio_synthetic as fs  # noqa: E402
from farlio_stream import compose, pose_error, Rig  # noqa: E402

CASE = os.environ.get('FARLIO_IP_CASE', 'told')
EXEC = os.environ.get('FARLIO_IP_EXEC', 'tam_gicp_ext')
CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'config', 'f1tenth.yml')
T_START = (2.0, -1.0, math.radians(30.0))                  # the car's start in the map frame
OFFSET = (0.1, 0.05, math.radians(2.0))   # inside the matcher's acceptance ellipsis (0.15 x 0.075 m): see the docstring
MAP_PCD = os.path.join(tempfile.mkdtemp(prefix='farlio_18xx_'), 'map.pcd')
# (position m, yaw deg) a lidar odometry may be off T_START after a correct pose: 2 x the worst of 10 runs, the
# measurement beside it. None until measured.
# Measured 2026-11-02, 10 runs each (load < 8), offset 0.1 / 0.05 m / 2 deg: FARLIO_1830 first lidar odometry 0.0796 m /
# 0.307 deg from T_START in all 10, worst of the 100 odometry 0.0991 m / 0.382 deg in all 10 (the run is deterministic
# to the 4th digit); FARLIO_1850 worst 0.0755 m / 0.297 deg in all 10. Bound = 2 x the worst, 0.0991 m / 0.382 deg.
BOUNDS_1830 = (0.20, 0.77)
REFUSALS = ('wait_tf_conflict', 'icp_cv_refused')
WITH_EKF = ('hold',)


@pytest.mark.launch_test
def generate_test_description():
    fs.write_map_pcd(MAP_PCD, T_START)
    tf = Node(package='tf2_ros', executable='static_transform_publisher',
              arguments=['--x', str(fs.LASER_X), '--z', str(fs.LASER_Z), '--frame-id', 'base_link',
                         '--child-frame-id', 'laser'])
    params = {'use_sim_time': True, 'pipeline.update_map': False, 'node.input_map': MAP_PCD,
              'node.input_initial_pose': '/core/state/initial_pose', 'node.wait_initial_pose': CASE != 'untold'}
    exe = EXEC
    if CASE == 'wait_tf_conflict':
        params['node.wait_tf'] = True
    if CASE == 'icp_cv_refused':
        exe = 'tam_icp_cv'
        params['node.input_initial_pose'] = '/core/state/initial_pose'
        params['node.wait_initial_pose'] = False
    odometry = Node(package='tam_odometry', executable=exe, namespace='/core/state', name='LidarOdometry',
                    output='screen', parameters=[CONFIG, params])
    actions = [tf, odometry]
    if CASE in WITH_EKF:
        actions.append(Node(package='tam_state_estimation_node', executable='tam_state_estimation_node_ekf_2d_kin',
                            namespace='/core/state', name='StateEstimation', output='screen',
                            parameters=[CONFIG, {'use_sim_time': True}]))
    return LaunchDescription(actions + [launch_testing.actions.ReadyToTest()]), {'odometry': odometry}


def window(rig, since_stamp):
    """Lidar odometry (x, y, yaw) recorded at or after sim time `since_stamp`."""
    return [(p[1], p[2], p[3]) for p in rig.lidar if p[0] >= since_stamp]


def check_bounds(test, poses, label):
    worst = max(pose_error(p, T_START)[0] for p in poses)
    worst_yaw = max(pose_error(p, T_START)[1] for p in poses)
    print('%s %d lidar odometry, worst %.4f m %.3f deg from T_START' % (label, len(poses), worst, worst_yaw))
    test.assertIsNotNone(BOUNDS_1830, 'bounds measured')
    test.assertLessEqual(worst, BOUNDS_1830[0], '%s: worst %.4f m from T_START' % (label, worst))
    test.assertLessEqual(worst_yaw, BOUNDS_1830[1], '%s: worst %.3f deg from T_START' % (label, worst_yaw))


class RigCase(unittest.TestCase):

    def setUp(self):
        self.rig = Rig()

    def tearDown(self):
        self.rig.close()


if CASE in ('told', 'untold'):
    class TestPoseIsFirstGuess(RigCase):

        def test_farlio_1830_pose_is_the_first_guess(self):
            rig = self.rig
            _, pose_sub = rig.wait_graph(pose_sub=CASE == 'told')
            if CASE == 'told':
                rig.publish_pose(compose(T_START, OFFSET))
                rig.drain(0.5)   # the scans start once the pose is in and settled
            t0 = rig.sim_time()
            rig.run_standing(1.0)
            rig.drain(0.5)
            poses = window(rig, t0)
            self.assertGreater(len(poses), 0, 'no lidar odometry (pose subscriber seen: %s)' % pose_sub)
            d, dyaw = pose_error(poses[0], T_START)
            if CASE == 'untold':
                self.assertGreaterEqual(d, 1.0, 'untold, the first lidar odometry is only %.2f m from T_START' % d)
                return
            self.assertLess(d, 1.0, 'first lidar odometry %.2f m from T_START' % d)
            print('FARLIO_1830 first lidar odometry %.4f m %.3f deg from T_START' % (d, dyaw))
            check_bounds(self, poses, 'FARLIO_1830')

if CASE == 'hold':
    class TestHold(RigCase):

        def test_farlio_1820_no_odometry_until_told(self):
            rig = self.rig
            rig.wait_graph(imu_sub=True)
            rig.run_standing(3.0)
            self.assertEqual(len(rig.lidar), 0, '%d lidar odometry messages during the hold' % len(rig.lidar))
            self.assertEqual(len(rig.ekf), 0, '%d EKF odometry messages during the hold' % len(rig.ekf))
            rig.publish_pose(compose(T_START, OFFSET))
            for _ in range(200):
                rig.tick(rig.frame(0.0, 0.0, 0.0))
                if rig.lidar and rig.ekf:
                    break
            self.assertGreater(len(rig.lidar), 0, 'no lidar odometry within 2 s of the pose')
            self.assertGreater(len(rig.ekf), 0, 'no EKF odometry within 2 s of the pose (lidar status: %s)' % rig.status_summary())

if CASE == 'wrong_frame':
    class TestWrongFrame(RigCase):

        def test_farlio_1840_wrong_frame_is_rejected(self, proc_output, odometry):
            rig = self.rig
            rig.wait_graph()
            rig.publish_pose(compose(T_START, OFFSET), frame='map')
            rig.run_standing(2.0)
            self.assertEqual(len(rig.lidar), 0, 'odometry during the hold after a wrong-frame pose')
            proc_output.assertWaitFor(re.compile(r"initial pose in frame 'map' ignored: expected 'far_lio_odom'"),
                                      process=odometry, timeout=5)

if CASE == 'late_pose':
    class TestLatePose(RigCase):

        def test_farlio_1850_late_pose_is_ignored(self, proc_output, odometry):
            rig = self.rig
            rig.wait_graph()
            rig.publish_pose(compose(T_START, OFFSET))
            rig.drain(0.5)
            rig.run_standing(1.0)
            self.assertGreater(len(rig.lidar), 0, 'no lidar odometry after the first pose')
            t1 = rig.sim_time()
            rig.publish_pose(compose(T_START, (2.0, 0.0, 0.0)))
            rig.run_standing(1.0)
            rig.drain(0.5)
            poses = window(rig, t1 + 0.05)
            self.assertGreater(len(poses), 0, 'no lidar odometry after the late pose')
            jump = max(pose_error(p, T_START)[0] for p in poses)
            self.assertLess(jump, 1.0, 'jump %.1f m after a late pose' % jump)
            check_bounds(self, poses, 'FARLIO_1850')
            proc_output.assertWaitFor(re.compile(r'relocalization not supported'), process=odometry, timeout=5)

if CASE in REFUSALS:
    class TestRefusal(unittest.TestCase):

        def test_farlio_1860_node_refuses_to_start(self, proc_output, proc_info, odometry):
            proc_info.assertWaitForShutdown(process=odometry, timeout=10)
            names = {'wait_tf_conflict': r'wait_initial_pose.*wait_tf|wait_tf.*wait_initial_pose',
                     'icp_cv_refused': r'input_initial_pose.*model|model.*input_initial_pose'}[CASE]
            proc_output.assertWaitFor(re.compile(names), process=odometry, timeout=5)

    @launch_testing.post_shutdown_test()
    class TestRefusalExit(unittest.TestCase):

        def test_farlio_1860_exit_is_not_zero(self, proc_info, odometry):
            self.assertNotEqual(proc_info[odometry].returncode, 0)
