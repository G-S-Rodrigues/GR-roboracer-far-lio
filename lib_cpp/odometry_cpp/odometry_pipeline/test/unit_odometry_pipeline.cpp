/*
 * Copyright 2026 Maximilian Leitenstern, Marcel Weinmann
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

#include <gtest/gtest.h>

#include <iostream>

#include "odometry_pipeline/odometry_pipeline.hpp"
#include "param_management_cpp/param_reference_manager.hpp"
#include "tsl_logger_cpp/reference_logger.hpp"

/**
 * @brief Test construction of OdometryPipeline from param manager and logger
 */
TEST(OdometryPipeline, BuildPmgLogger)
{
  // clang-format off
  // Construct from param manager and logger
  tam::pmg::ParamReferenceManager::UniquePtr pmg_ =  std::make_unique<tam::pmg::ParamReferenceManager>(); // NOLINT
  tam::tsl::ReferenceLogger::UniquePtr logger_ = std::make_unique<tam::tsl::ReferenceLogger>(); // NOLINT

  std::unique_ptr<tam::core::state::OdometryPipeline<tam::core::state::types::ICP_EXT>> odom_pipeline =  // NOLINT
    tam::core::state::OdometryPipeline<tam::core::state::types::ICP_EXT>::from_config(pmg_.get(), logger_.get()); // NOLINT
  // clang-format on

  // Check correct parameterization based on single parameter
  tam::pmg::MgmtInterface* pmg_raw = pmg_.get();
  pmg_raw->set_value("diagnostic.frame_outdated", 97.0);  // NOLINT
  EXPECT_EQ(odom_pipeline->get_diagnostic_config().frame_outdated, 97.0);
}

/**
 * @brief Test construction of OdometryPipeline from config and debug object
 */
TEST(OdometryPipeline, BuildConfigDebug)
{
  // Construct from config and debug object
  tam::core::state::types::PipelineConfig config;
  tam::core::state::types::PipelineDebug debug;
  // Check correct parameterization based on single parameter

  std::unique_ptr<tam::core::state::OdometryPipeline<tam::core::state::types::ICP_EXT>> odom_pipeline =  // NOLINT
    tam::core::state::OdometryPipeline<tam::core::state::types::ICP_EXT>::from_config(config, debug);    // NOLINT

  // Check correct parameterization based on single parameter
  odom_pipeline->get_diagnostic_config().frame_outdated = 97.0;

  EXPECT_EQ(odom_pipeline->get_diagnostic_config().frame_outdated, 97.0);
}

// ---- FARLIO-1310..1330: pipeline.lidar_2d ----
namespace {
template <typename TConfig>
std::unique_ptr<tam::core::state::OdometryPipeline<TConfig>> make_2d_pipeline(
  tam::pmg::ParamReferenceManager::UniquePtr& pmg, tam::tsl::ReferenceLogger::UniquePtr& logger)
{
  pmg = std::make_unique<tam::pmg::ParamReferenceManager>();
  logger = std::make_unique<tam::tsl::ReferenceLogger>();
  return tam::core::state::OdometryPipeline<TConfig>::from_config(pmg.get(), logger.get());
}
}  // namespace

TEST(FARLIO_1310, Lidar2dDefaultsOff)
{
  using TConfig = tam::core::state::types::GICP_EXT;
  tam::pmg::ParamReferenceManager::UniquePtr pmg;
  tam::tsl::ReferenceLogger::UniquePtr logger;
  auto pipeline = make_2d_pipeline<TConfig>(pmg, logger);
  pipeline->init();
  EXPECT_FALSE(pipeline->get_map_config().planar);
  EXPECT_FALSE(pipeline->get_registration_config().planar);
  EXPECT_FALSE(pipeline->get_frame_map_config().planar);
}

TEST(FARLIO_1320, Lidar2dReachesMapFrameMapAndRegistration)
{
  using TConfig = tam::core::state::types::GICP_EXT;
  tam::pmg::ParamReferenceManager::UniquePtr pmg;
  tam::tsl::ReferenceLogger::UniquePtr logger;
  auto pipeline = make_2d_pipeline<TConfig>(pmg, logger);
  static_cast<tam::pmg::MgmtInterface*>(pmg.get())->set_value("pipeline.lidar_2d", true);
  pipeline->init();
  EXPECT_TRUE(pipeline->get_map_config().planar);
  EXPECT_TRUE(pipeline->get_registration_config().planar);
  EXPECT_TRUE(pipeline->get_frame_map_config().planar);
}

TEST(FARLIO_1330, Lidar2dRegisteredPoseKeepsZAndLevelAttitude)
{
  using TConfig = tam::core::state::types::GICP_EXT;
  for (const bool buffer_z : {false, true}) {
    tam::pmg::ParamReferenceManager::UniquePtr pmg;
    tam::tsl::ReferenceLogger::UniquePtr logger;
    auto pipeline = make_2d_pipeline<TConfig>(pmg, logger);
    tam::pmg::MgmtInterface* pmg_if = pmg.get();
    pmg_if->set_value("pipeline.lidar_2d", true);
    pmg_if->set_value("pipeline.update_map", true);
    pmg_if->set_value("pipeline.undistort", false);
    pmg_if->set_value("pipeline.preprocess", false);
    pmg_if->set_value("pipeline.downsample", false);
    pmg_if->set_value("pipeline.buffer_z", buffer_z);
    pmg_if->set_value("map.voxel_size", 0.5);
    pipeline->init();

    // a square room outline at z = 0, 0.1 m spacing
    std::vector<tam::core::state::types::Point<TConfig>> frame;
    auto add = [&](float x, float y) {
      tam::core::state::types::Point<TConfig> p{};
      p.pos = Eigen::Vector3f(x, y, 0.0f);
      frame.push_back(p);
    };
    for (int i = 0; i <= 80; ++i) {
      const float t = -4.0f + 0.1f * static_cast<float>(i);
      add(t, -4.0f);
      add(t, 4.0f);
      add(-4.0f, t);
      add(4.0f, t);
    }
    // frame 0 seeds the map at the identity
    tam::core::state::types::PoseStamped ps{};
    ps.stamp = 1000000000ull;
    ps.pose = Sophus::SE3f();
    pipeline->set_pose_model(ps, true);
    const auto odom0 = pipeline->register_frame(frame, ps.stamp);
    const bool frame0_ok = odom0.status.level == tam::core::state::types::DiagnosticLevel::OK;
    std::printf("FARLIO_1330 buffer_z %d: frame 0 status ok %d\n", buffer_z, frame0_ok);

    // second frame: the guess carries 1 deg of roll and pitch and z = 0.2
    ps.stamp += 100000000ull;
    const float deg = static_cast<float>(M_PI) / 180.0f;
    ps.pose = Sophus::SE3f(
      Eigen::Quaternionf(Eigen::AngleAxisf(deg, Eigen::Vector3f::UnitX()) *
                         Eigen::AngleAxisf(deg, Eigen::Vector3f::UnitY())),
      Eigen::Vector3f(0.02f, 0.0f, 0.2f));
    pipeline->set_pose_model(ps, true);
    const auto odom = pipeline->register_frame(frame, ps.stamp);
    const Sophus::SE3f& pose = odom.pose.pose;
    // z is the guess's z, or the buffered z (frame 0's) under buffer_z when frame 0 was OK
    EXPECT_FLOAT_EQ(pose.translation().z(), (buffer_z && frame0_ok) ? odom0.pose.pose.translation().z() : 0.2f) << "buffer_z " << buffer_z;
    EXPECT_EQ(pose.unit_quaternion().x(), 0.0f) << "buffer_z " << buffer_z;
    EXPECT_EQ(pose.unit_quaternion().y(), 0.0f) << "buffer_z " << buffer_z;
  }
}
