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

// FARLIO-1100: how many points of a single 64-beam (and 1080-beam) planar frame get a non-zero
// covariance in a frame VoxelHashMap<GICP_EXT>, at voxel sizes 0.25 / 0.5 / 1.0 m. This is the
// screen for the plan's one unproven assumption; FARLIO-1210 (accuracy) is the real trigger.
//
// Measured on 2026-10-25 (CPU, Release, 12 poses on the stadium oval, noise 3 cm, seed 20261025,
// FROBENIUS), mean / min fraction of frame points with a non-zero covariance:
//     beams  voxel   mean   min
//        64   0.25  0.000  0.000   (no voxel holds 5 neighbours: NUM_NEIGHBORS is 5)
//        64   0.50  0.605  0.562   <- smallest voxel that passes on every pose; step 1b's pick
//        64   1.00  0.697  0.636
//      1080   0.25  0.921  0.788
//      1080   0.50  0.944  0.827
//      1080   1.00  0.950  0.910
#include <gtest/gtest.h>

#include <cstdio>
#include <vector>

#include "f1tenth_track.hpp"
#include "map_handler/voxel_hash_map.hpp"

namespace
{
using tam::core::state::types::GICP_EXT;
using tam::core::state::types::Point;
using tam::core::state::types::Point_XYZ;

constexpr std::uint32_t kSeed = 20261025;
constexpr int kNumPoses = 12;
constexpr double kDensityGate = 0.5;  // plan: at least 50 % of 64-beam frame points

struct Density {
  double mean;
  double min;
};

Density frame_density(const int num_beams, const double voxel)
{
  tam::core::state::types::MapConfig config;
  tam::core::state::types::MapDebug debug;
  config.voxel_size = voxel;
  config.max_distance = 50.0;
  config.frame_map = true;
  config.cov_regularization = "FROBENIUS";
  auto map = tam::core::state::VoxelHashMap<GICP_EXT>::from_config(config, debug);
  map->init();

  farlio_f1tenth::Noise noise(kSeed);
  const double lap = 2.0 * farlio_f1tenth::kStraight + 2.0 * farlio_f1tenth::kPi * farlio_f1tenth::kRadius;
  double sum = 0.0;
  double min = 1.0;
  for (int p = 0; p < kNumPoses; ++p) {
    const auto pose = farlio_f1tenth::centerline_pose(lap * p / kNumPoses);
    std::vector<Point<Point_XYZ>> frame;
    for (const auto & h : farlio_f1tenth::cast_scan(pose, num_beams, &noise)) {
      Point<Point_XYZ> pt;
      pt.pos = h;
      frame.push_back(pt);
    }
    map->add_points(frame);
    const auto cloud = map->get_cloud();
    std::size_t with_cov = 0;
    for (const auto & pt : cloud)
      if (pt.cov.norm() > 0.0f) ++with_cov;
    const double frac = cloud.empty() ? 0.0 : static_cast<double>(with_cov) / cloud.size();
    sum += frac;
    min = std::min(min, frac);
  }
  return {sum / kNumPoses, min};
}
}  // namespace

TEST(FARLIO_1100, PlanarFrameDensity)
{
  std::printf("FARLIO_1100 frame points with non-zero covariance (mean / min over %d poses)\n", kNumPoses);
  std::printf("FARLIO_1100 beams  voxel   mean    min\n");
  for (const int beams : {64, 1080}) {
    for (const double voxel : {0.25, 0.5, 1.0}) {
      const Density d = frame_density(beams, voxel);
      std::printf("FARLIO_1100 %5d  %.2f  %.3f  %.3f\n", beams, voxel, d.mean, d.min);
    }
  }
  // The gate: at least one voxel size gives >= 50 % at 64 beams.
  double best = 0.0;
  for (const double voxel : {0.25, 0.5, 1.0}) best = std::max(best, frame_density(64, voxel).mean);
  EXPECT_GE(best, kDensityGate) << "no voxel size gives >= 50 % covariance density at 64 beams";
}
