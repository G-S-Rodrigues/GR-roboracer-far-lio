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
#pragma once
// FARLIO-1200..1240: the planar (SE(2)) solve on the synthetic F1TENTH track. Included by
// unit_gicp.cpp and unit_icp.cpp. (FARLIO-1250 is the step-1c golden staying bit-exact.)
// Host (CPU) only: the CUDA twins are not written yet.
#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

#ifndef __CUDACC__
#include "f1tenth_track.hpp"
#include "test_utils.hpp"

namespace farlio_planar_solve
{
namespace types = tam::core::state::types;

struct Result {
  Sophus::SE3f pose;
  Sophus::SE3f truth;
};

constexpr double kVoxel = 0.5;  // step 1b's pick (unit_planar_density.cpp)

/// Register one scan from the track's pose `s` against the wall map, starting from a guess offset
/// from the truth. `planar` switches both the map covariance and the solver.
template <typename TConfig, template <typename> class Reg>
Result register_scan(const std::string & solver, const int beams, const bool planar, const Sophus::SE3f & offset,
  const double s = 3.0, const double z_sigma = 0.0)
{
  types::MapConfig map_config;
  types::MapDebug map_debug;
  map_config.voxel_size = kVoxel;
  map_config.max_distance = 100.0;
  map_config.cov_regularization = "FROBENIUS";
  map_config.planar = planar;
  auto map = tam::core::state::VoxelHashMap<TConfig>::from_config(map_config, map_debug);
  map->init();
  farlio_f1tenth::Noise z_map_noise(777);
  std::vector<types::Point<types::Point_XYZ>> wall;
  for (const auto & p : farlio_f1tenth::wall_points()) {
    types::Point<types::Point_XYZ> pt;
    pt.pos = p;
    pt.pos.z() = static_cast<float>(z_sigma * z_map_noise.gaussian());
    wall.push_back(pt);
  }
  map->add_points(wall);

  types::RegistrationConfig config;
  types::RegistrationDebug debug;
  config.solver_type = solver;
  config.max_iter = 100;
  config.convergence_criterion = 5.0e-3;
  config.damping_factor = 1.0e-9;
  config.damping_scale = 2.0;
  config.planar = planar;
  auto registration = Reg<TConfig>::from_config(config, debug);
  registration->init();
  if constexpr (types::FRAMEMAP<TConfig>) registration->init_frame_map(map->get_config());

  const auto pose2d = farlio_f1tenth::centerline_pose(s);
  const Sophus::SE3f truth(Eigen::Quaternionf(Eigen::AngleAxisf(static_cast<float>(pose2d.yaw), Eigen::Vector3f::UnitZ())),
    Eigen::Vector3f(static_cast<float>(pose2d.x), static_cast<float>(pose2d.y), 0.0f));
  farlio_f1tenth::Noise noise(20261025);
  farlio_f1tenth::Noise z_frame_noise(778);
  std::vector<types::Point<TConfig>> frame;
  for (const auto & h : farlio_f1tenth::cast_scan(pose2d, beams, &noise)) {
    types::Point<TConfig> pt{};
    pt.pos = h;
    pt.pos.z() = static_cast<float>(z_sigma * z_frame_noise.gaussian());
    frame.push_back(pt);
  }
  const Sophus::SE3f guess = truth * offset;
  const Sophus::SE3f start = planar ? tam::core::state::utils::project_se2(guess) : guess;
  return {registration->register_frame(frame, map.get(), start, 1.0f, 0.1f), truth};
}

inline Sophus::SE3f offset_xy_yaw(const float x, const float y, const float yaw_deg)
{
  return Sophus::SE3f(
    Eigen::Quaternionf(Eigen::AngleAxisf(yaw_deg * static_cast<float>(M_PI) / 180.0f, Eigen::Vector3f::UnitZ())),
    Eigen::Vector3f(x, y, 0.0f));
}

inline float rot_err_deg(const Sophus::SE3f & a, const Sophus::SE3f & b)
{
  return (a.so3().inverse() * b.so3()).log().norm() * 180.0f / static_cast<float>(M_PI);
}
}  // namespace farlio_planar_solve

/// FARLIO-1200: project_se2 keeps yaw, x, y and z and zeroes roll and pitch.
TEST(FARLIO_1200, ProjectSe2)
{
  const Eigen::Quaternionf q = Eigen::AngleAxisf(0.7f, Eigen::Vector3f::UnitZ()) *
                               Eigen::AngleAxisf(0.05f, Eigen::Vector3f::UnitY()) *
                               Eigen::AngleAxisf(-0.03f, Eigen::Vector3f::UnitX());
  const Sophus::SE3f p(q, Eigen::Vector3f(1.0f, 2.0f, 0.031f));
  const Sophus::SE3f r = tam::core::state::utils::project_se2(p);
  EXPECT_EQ(r.translation(), p.translation());
  EXPECT_EQ(r.unit_quaternion().x(), 0.0f);
  EXPECT_EQ(r.unit_quaternion().y(), 0.0f);
  const Eigen::Vector3f fwd = p.unit_quaternion() * Eigen::Vector3f::UnitX();
  EXPECT_NEAR(2.0f * std::atan2(r.unit_quaternion().z(), r.unit_quaternion().w()), std::atan2(fwd.y(), fwd.x()), 1e-5f);
}

/// FARLIO-1210: from a (0.10 m, -0.08 m, 2 deg) offset, planar on, both solvers, 1080 and 64 beams.
template <typename TConfig, template <typename> class Reg>
void run_1210()
{
  for (const std::string solver : {"GaussNewton", "LevenbergMarquardt"}) {
    for (const int beams : {1080, 64}) {
      const auto r = farlio_planar_solve::register_scan<TConfig, Reg>(
        solver, beams, true, farlio_planar_solve::offset_xy_yaw(0.10f, -0.08f, 2.0f));
      const float terr = (r.pose.translation() - r.truth.translation()).norm();
      const float rerr = farlio_planar_solve::rot_err_deg(r.pose, r.truth);
      std::printf("FARLIO_1210 %s %d beams: %.4f m %.3f deg\n", solver.c_str(), beams, terr, rerr);
      EXPECT_LE(terr, 0.01f) << solver << " " << beams;
      EXPECT_LE(rerr, 0.2f) << solver << " " << beams;
    }
  }
}

/// FARLIO-1220 / 1230: planar on leaves z, roll and pitch at the (projected) guess, tilted or not.
template <typename TConfig, template <typename> class Reg>
void run_1220_1230()
{
  for (const std::string solver : {"GaussNewton", "LevenbergMarquardt"}) {
    for (const float roll_deg : {0.0f, 1.0f}) {
      Sophus::SE3f off = farlio_planar_solve::offset_xy_yaw(0.10f, -0.08f, 2.0f);
      off = Sophus::SE3f(
        off.unit_quaternion() * Eigen::Quaternionf(Eigen::AngleAxisf(
                                  roll_deg * static_cast<float>(M_PI) / 180.0f, Eigen::Vector3f::UnitX())),
        off.translation());
      const auto r = farlio_planar_solve::register_scan<TConfig, Reg>(solver, 1080, true, off);
      EXPECT_EQ(r.pose.translation().z(), 0.0f) << solver << " roll " << roll_deg;
      EXPECT_EQ(r.pose.unit_quaternion().x(), 0.0f) << solver << " roll " << roll_deg;
      EXPECT_EQ(r.pose.unit_quaternion().y(), 0.0f) << solver << " roll " << roll_deg;
    }
  }
}

/// FARLIO-1240: planar off: z / roll / pitch drift is non-zero (why the flag exists).
/// The scene has 1 cm of height noise on the map walls and on the scan hits (seeded). A noise-free
/// coplanar scene gives exactly 0 drift (measured, GICP and ICP): every normal lies in the plane, so the
/// z / roll / pitch rows of the update vanish, and the premise only holds for a scene with height
/// structure, which a real wall is. Measured drift at sigma 0.005 / 0.01 / 0.03 m: GICP 4.8e-4 / 1.1e-3 /
/// 7.4e-3, ICP 8.8e-4 / 2.1e-3 / 5.7e-3; at sigma 0 it is 0.
template <typename TConfig, template <typename> class Reg>
void run_1240()
{
  const auto r = farlio_planar_solve::register_scan<TConfig, Reg>(
    "GaussNewton", 1080, false, farlio_planar_solve::offset_xy_yaw(0.10f, -0.08f, 2.0f), 3.0, 0.01);
  const float drift = std::abs(r.pose.translation().z()) + std::abs(r.pose.unit_quaternion().x()) +
                      std::abs(r.pose.unit_quaternion().y());
  std::printf("FARLIO_1240 planar off drift %.3e\n", drift);
  EXPECT_GT(drift, 1.0e-6f);
}
#endif  // __CUDACC__
