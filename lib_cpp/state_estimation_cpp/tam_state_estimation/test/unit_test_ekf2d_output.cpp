/*
 * Copyright 2026 Guilherme Rodrigues
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// FARLIO_1620: the 2D filters' odometry output carries z = 0.
//
// get_odometry() copies x, y, yaw and the planar velocity into a stack-local Odometry and never
// writes position_m.z, so a 2D filter published whatever the stack held there (6.95e-310 seen in
// the lap test). This target is built with -ftrivial-auto-var-init=pattern, which fills every
// uninitialised automatic variable with 0xFE bytes, so the unwritten z is a fixed non-zero value
// here and the test is deterministic rather than dependent on stack garbage.
#include <gtest/gtest.h>

#include "state_estimation_constants/EKF_2D.hpp"
#include "tam_state_estimation/state_estimation.hpp"
#include "tum_types_cpp/common.hpp"

TEST(Ekf2dOutput, FARLIO_1620_PositionZIsZero)
{
  using tam::core::state::EKF_2D;
  using tam::core::state::VehicleModel;
  using Filter = tam::core::state::StateEstimation<EKF_2D, VehicleModel::NonHolonomic>;
  Filter StateEstimation;

  StateEstimation.set_input_position_status(tam::types::ErrorLvl::OK, 0);
  StateEstimation.set_input_orientation_status(tam::types::ErrorLvl::OK, 0);
  StateEstimation.set_input_imu_status(tam::types::ErrorLvl::OK, 0);
  StateEstimation.set_input_imu_status(tam::types::ErrorLvl::OK, 1);
  StateEstimation.set_input_wheelspeed_status(tam::types::ErrorLvl::OK);

  tam::types::control::Odometry input_odometry{};
  input_odometry.position_m.x = 1.5;
  input_odometry.position_m.y = -2.5;
  StateEstimation.set_input_position(input_odometry, 0);
  StateEstimation.set_input_orientation(input_odometry, 0, true, true, true);
  tam::types::common::DataPerWheel<double> input_wheelspeed{};
  StateEstimation.set_input_wheelspeeds(input_wheelspeed);
  StateEstimation.set_initial_state();
  StateEstimation.step();

  const tam::types::control::Odometry out = StateEstimation.get_odometry();
  EXPECT_EQ(out.position_m.z, 0.0);
}
