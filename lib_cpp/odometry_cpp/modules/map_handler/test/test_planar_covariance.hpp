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
// FARLIO-1110 / 1120: planar covariance in the voxel hash map. Included by unit_voxel_hash_map.cpp
// and .cu. (FARLIO-1130, the frame map inheriting the flag, lives in registration_handler/test,
// because map_handler does not link registration_handler.)
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "test_utils.hpp"

namespace farlio_planar
{
using tam::core::state::types::Point;
using tam::core::state::types::Point_XYZ;

/// Build a map holding `pts` (input points, Point_XYZ) with the given regularizer and planar flag.
template <typename TConfig>
std::vector<Point<TConfig>> map_cloud(
  const std::vector<Eigen::Vector3f> & pts, const std::string & cov_reg, const bool planar)
{
  tam::core::state::types::MapConfig config;
  tam::core::state::types::MapDebug debug;
  config.voxel_size = 1.0;
  config.max_distance = 50.0;
  config.cov_regularization = cov_reg;
  config.planar = planar;
#ifdef __CUDACC__
  auto map = tam::core::state::cuda::VoxelHashMap<TConfig>::from_config(config, debug);
#else
  auto map = tam::core::state::VoxelHashMap<TConfig>::from_config(config, debug);  // NOLINT
#endif
  map->init();
  std::vector<Point<Point_XYZ>> frame;
  for (const auto & p : pts) {
    Point<Point_XYZ> pt;
    pt.pos = p;
    frame.push_back(pt);
  }
  map->add_points(frame);
#ifdef __CUDACC__
  const auto d = map->get_cloud();
  std::vector<Point<TConfig>> cloud(d.size());
  thrust::copy(d.begin(), d.end(), cloud.begin());
  return cloud;
#else
  return map->get_cloud();
#endif
}

/// A horizontal wall line at z = 0: points along x at y = 2, 0.2 m apart (9 points in two voxels).
inline std::vector<Eigen::Vector3f> wall_line()
{
  std::vector<Eigen::Vector3f> pts;
  for (int i = 0; i < 9; ++i) pts.emplace_back(0.2f * i, 2.0f, 0.0f);
  return pts;
}
}  // namespace farlio_planar

/// FARLIO-1110: with `planar` on, the normal of a horizontal wall is in the plane and perpendicular
/// to the wall, for each regularizer (it flips to z for SVD without the extrusion).
TEST(FARLIO_1110, PlanarNormalInPlane)
{
  using TConfig = tam::core::state::types::POINT_NORMAL;
  for (const std::string reg : {"SVD", "FROBENIUS", "MIN_EIGENVALUE"}) {
    const auto cloud = farlio_planar::map_cloud<TConfig>(farlio_planar::wall_line(), reg, true);
    int with_normal = 0;
    for (const auto & p : cloud) {
      if (p.normal.norm() < 0.5f) continue;  // too few neighbours: zero normal by design
      ++with_normal;
      EXPECT_LT(std::abs(p.normal.z()), 1e-3f) << reg;
      EXPECT_LT(std::abs(p.normal.x()), 1e-3f) << reg;  // perpendicular to the wall (along x)
      EXPECT_GT(std::abs(p.normal.y()), 0.999f) << reg;
    }
    EXPECT_GT(with_normal, 0) << reg << ": no point got a normal";
  }
}

/// FARLIO-1120: with `planar` off the covariance is the upstream computation (the values below are
/// upstream's own expectations from the `Covariances` test, for the point (5.1, 2.1, 3.1)).
TEST(FARLIO_1120, PlanarOffIsUpstream)
{
  using TConfig = tam::core::state::types::POINT_NORMAL;
  const std::vector<Eigen::Vector3f> pts{{5.1f, 2.1f, 3.1f}, {5.8f, 2.5f, 3.9f}, {5.9f, 2.9f, 3.9f},
    {4.1f, 1.0f, 4.0f}, {4.2f, 1.1f, 3.0f}, {6.3f, 3.2f, 3.1f}, {5.4f, 2.3f, 4.2f}, {5.5f, 2.4f, 2.3f},
    {5.6f, 2.5f, 4.4f}, {5.7f, 2.6f, 2.5f}};
  Eigen::Matrix3f svd, frob, mine;
  svd << 0.685181, 0.463466, 0.0243394, 0.463466, 0.317701, -0.0358316, 0.0243394, -0.0358316, 0.998118;
  frob << 31.1507, 20.4267, 1.02138, 20.4267, 15.6264, -14.2987, 1.02138, -14.2987, 286.484;
  mine << 0.060013, 0.0399809, 0.00199909, 0.0399809, 0.0296282, -0.0279985, 0.00199909, -0.0279985, 0.560000;
  const std::vector<std::pair<std::string, Eigen::Matrix3f>> cases{{"SVD", svd}, {"FROBENIUS", frob},
    {"MIN_EIGENVALUE", mine}};
  for (const auto & [reg, expected] : cases) {
    const auto cloud = farlio_planar::map_cloud<TConfig>(pts, reg, false);
    bool found = false;
    for (const auto & p : cloud) {
      if ((p.pos - Eigen::Vector3f(5.1f, 2.1f, 3.1f)).norm() > 1e-5f) continue;
      found = true;
      EXPECT_TRUE(p.cov.isApprox(expected, 1e-5f)) << reg;
    }
    EXPECT_TRUE(found) << reg;
  }
}
