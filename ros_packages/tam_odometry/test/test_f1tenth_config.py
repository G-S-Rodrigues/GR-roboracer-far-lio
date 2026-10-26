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
"""FARLIO_1510: config/f1tenth.yml loads in both nodes of the F1TENTH pair.

tam_gicp_ext (2D lidar, LaserScan input) and tam_state_estimation_node_ekf_2d_kin are launched with
config/f1tenth.yml as their parameter file. Both must reach "Node initialized!" with no parameter
exception. Only use_sim_time is overridden (to false, so no /clock is needed); a static TF base_link ->
laser stands in for the sim's.
"""
import os
import time
import unittest

from launch import LaunchDescription
from launch_ros.actions import Node
import launch_testing
import launch_testing.actions
import launch_testing.asserts
import pytest
import rclpy

CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'config', 'f1tenth.yml')


@pytest.mark.launch_test
def generate_test_description():
    params = [CONFIG, {'use_sim_time': False}]
    tf = Node(package='tf2_ros', executable='static_transform_publisher',
              arguments=['--x', '0.10355', '--z', '0.031', '--frame-id', 'base_link', '--child-frame-id', 'laser'])
    odometry = Node(package='tam_odometry', executable='tam_gicp_ext', namespace='/core/state',
                    name='LidarOdometry', parameters=params, output='screen')
    ekf = Node(package='tam_state_estimation_node', executable='tam_state_estimation_node_ekf_2d_kin',
               namespace='/core/state', name='StateEstimation', parameters=params, output='screen')
    return (LaunchDescription([tf, odometry, ekf, launch_testing.actions.ReadyToTest()]),
            {'odometry': odometry, 'ekf': ekf})


class TestF1tenthConfig(unittest.TestCase):

    def test_farlio_1510_both_nodes_initialize(self, proc_output, proc_info, odometry, ekf):
        proc_output.assertWaitFor('Node initialized', process=odometry, timeout=60)
        # the EKF logs nothing at startup: it is up when it is on the graph and its process is still running
        rclpy.init()
        node = rclpy.create_node('f1tenth_config_probe')
        try:
            end = time.time() + 30.0
            names = set()
            while time.time() < end:
                names = set(node.get_node_names_and_namespaces())
                if ('StateEstimation', '/core/state') in names:
                    break
                rclpy.spin_once(node, timeout_sec=0.2)
            self.assertIn(('StateEstimation', '/core/state'), names)
            self.assertIn(('LidarOdometry', '/core/state'), names)
        finally:
            node.destroy_node()
            rclpy.shutdown()
        # a parameter exception ends a node: neither process may have exited
        with self.assertRaises(AssertionError):
            proc_info.assertWaitForShutdown(process=ekf, timeout=3)
        with self.assertRaises(AssertionError):
            proc_info.assertWaitForShutdown(process=odometry, timeout=1)
        for bad in ('InvalidParameter', 'ParameterUninitialized', 'terminate called', 'what():'):
            self.assertFalse(proc_output.waitFor(bad, timeout=0.5), bad)
