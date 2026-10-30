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
#pragma once
// FARLIO-1200..1240: the planar (SE(2)) solve on the synthetic F1TENTH track. Included by
// unit_gicp.cpp and unit_icp.cpp. (FARLIO-1250 is the step-1c golden staying bit-exact.)
// Host (CPU) only: the CUDA twins are not written yet.
#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
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

/// FARLIO-1210: from a (0.10 m, -0.08 m, 2 deg) offset, planar on, GaussNewton and LevenbergMarquardt (they
/// give identical results, so one bound covers both), 1080 and 64 beams, at positions on the track.
///
/// What one scan observes. Between parallel walls the along-track position is unobservable: sliding the
/// scan along the corridor leaves every range unchanged (repo gotcha #18, in its straight-track form). On a
/// straight the test therefore asserts only the lateral error and the heading, and prints the along-track
/// error without asserting it (measured 0.09-0.13 m, about the 0.10 m guess offset left uncorrected, except
/// GICP 1080 at 0.012 / 0.029 m). On a curve the wall bends and the whole pose error is asserted.
///
/// Bounds. The plan's 1 cm / 0.2 deg is asserted wherever the measurement meets it (each component
/// separately). Every other component has a bound of 2x the worst measured, rounded up to 1 mm / 1 mdeg, with
/// the measurement beside it (owner ruling, 2026-10-28). "Position" is the lateral error on a straight and
/// the total error on a curve. Measured 2026-10-28, deterministic (fixed noise seed, same result on rerun).
/// The loose entries (ICP on curves: 0.17 m / 2-3.5 deg) are the matcher's real accuracy at that position,
/// not a tolerance chosen for the test; the bite check in the hand-off records what they do and do not catch.
struct Bound1210 {
  double s;
  int beams;
  float pos_m;
  float rot_deg;
};
inline bool is_straight(const double s)
{
  return s < 8.0 || (s >= 8.0 + 3.0 * M_PI + 1.0e-9 && s < 16.0 + 3.0 * M_PI);
}

constexpr Bound1210 kGicp1210[] = {
  {3.0, 1080, 0.010f, 0.200f},  // measured 0.0026 m 0.129 deg
  {3.0, 64, 0.010f, 0.638f},  // measured 0.0048 m 0.319 deg
  {20.0, 1080, 0.010f, 0.200f},  // measured 0.0004 m 0.178 deg
  {20.0, 64, 0.020f, 1.254f},  // measured 0.0100 m 0.627 deg
  {10.0, 1080, 0.071f, 1.702f},  // measured 0.0354 m 0.851 deg
  {10.0, 64, 0.111f, 1.900f},  // measured 0.0554 m 0.950 deg
  {14.0, 1080, 0.010f, 0.200f},  // measured 0.0095 m 0.036 deg
  {14.0, 64, 0.115f, 1.594f},  // measured 0.0574 m 0.797 deg
  {30.0, 1080, 0.048f, 1.002f},  // measured 0.0237 m 0.501 deg
  {30.0, 64, 0.177f, 3.102f},  // measured 0.0881 m 1.551 deg
};
constexpr Bound1210 kIcp1210[] = {
  {3.0, 1080, 0.010f, 1.084f},  // measured 0.0034 m 0.542 deg
  {3.0, 64, 0.010f, 0.912f},  // measured 0.0052 m 0.456 deg
  {20.0, 1080, 0.010f, 1.046f},  // measured 0.0036 m 0.523 deg
  {20.0, 64, 0.010f, 1.542f},  // measured 0.0081 m 0.771 deg
  {10.0, 1080, 0.163f, 2.816f},  // measured 0.0813 m 1.408 deg
  {10.0, 64, 0.158f, 2.592f},  // measured 0.0786 m 1.296 deg
  {14.0, 1080, 0.170f, 1.980f},  // measured 0.0848 m 0.990 deg
  {14.0, 64, 0.169f, 2.060f},  // measured 0.0843 m 1.030 deg
  {30.0, 1080, 0.170f, 2.794f},  // measured 0.0850 m 1.397 deg
  {30.0, 64, 0.204f, 3.552f},  // measured 0.1017 m 1.776 deg
};

template <typename TConfig, template <typename> class Reg, std::size_t N>
void run_1210(const Bound1210 (&bounds)[N])
{
  for (const auto & bound : bounds) {
    for (const std::string solver : {"GaussNewton", "LevenbergMarquardt"}) {
      const auto r = farlio_planar_solve::register_scan<TConfig, Reg>(
        solver, bound.beams, true, farlio_planar_solve::offset_xy_yaw(0.10f, -0.08f, 2.0f), bound.s);
      // Error in the truth's body frame: x along the track, y across it.
      const Eigen::Vector3f e = r.truth.so3().inverse() * (r.pose.translation() - r.truth.translation());
      const bool straight = is_straight(bound.s);
      const float pos = straight ? std::abs(e.y()) : std::hypot(e.x(), e.y());
      const float rerr = farlio_planar_solve::rot_err_deg(r.pose, r.truth);
      std::printf("FARLIO_1210 s %.0f %s %s %d beams: along %.4f lateral %.4f m, asserted %.4f m (<= %.3f), %.3f deg (<= %.3f)\n",
        bound.s, straight ? "straight" : "curve", solver.c_str(), bound.beams, e.x(), e.y(), pos, bound.pos_m, rerr,
        bound.rot_deg);
      EXPECT_LE(pos, bound.pos_m) << "s " << bound.s << " " << solver << " " << bound.beams;
      EXPECT_LE(rerr, bound.rot_deg) << "s " << bound.s << " " << solver << " " << bound.beams;
      // Beats its start (review B1). Several bounds above exceed the start offset, so a matcher that returns its
      // guess would pass them. The start error is the same metric of the start offset: the lateral part on a
      // straight, the total on a curve, and the rotation of the offset. Every measured row is below it (worst
      // ICP s=30 64 beams: 0.1017 m vs 0.128 m, 1.776 deg vs 2.000 deg).
      const Sophus::SE3f start_offset = farlio_planar_solve::offset_xy_yaw(0.10f, -0.08f, 2.0f);
      const float start_pos = straight ? std::abs(start_offset.translation().y())
                                       : std::hypot(start_offset.translation().x(), start_offset.translation().y());
      const float start_rot = farlio_planar_solve::rot_err_deg(start_offset, Sophus::SE3f());
      EXPECT_LT(pos, start_pos - 1.0e-3f) << "no better than its start, s " << bound.s << " " << solver << " " << bound.beams;
      EXPECT_LT(rerr, start_rot - 1.0e-3f) << "no better than its start, s " << bound.s << " " << solver << " " << bound.beams;
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
