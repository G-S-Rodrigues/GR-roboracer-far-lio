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

// FARLIO-1710: CPU registration latency per scan on the synthetic F1TENTH track, planar (SE(2)) solve,
// 64 and 1080 beams, GICP_EXT and ICP_EXT. Reported, not gated (spec acceptance 7; Q3 target <= 10 ms/scan).
// Usage: bench_planar_example [scans_per_case]   (default 200)
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

#include "../test/f1tenth_track.hpp"
#include "map_handler/voxel_hash_map.hpp"
#include "odometry_utils/utils.hpp"
#include "registration_handler/gicp.hpp"
#include "registration_handler/icp.hpp"

namespace types = tam::core::state::types;

template <typename TConfig, template <typename> class Reg>
void bench(const char * name, const int beams, const int scans)
{
  types::MapConfig map_config;
  types::MapDebug map_debug;
  map_config.voxel_size = 0.5;  // step 1b / config/f1tenth.yml
  map_config.max_distance = 100.0;
  map_config.cov_regularization = "FROBENIUS";
  map_config.planar = true;
  auto map = tam::core::state::VoxelHashMap<TConfig>::from_config(map_config, map_debug);
  map->init();
  std::vector<types::Point<types::Point_XYZ>> wall;
  for (const auto & p : farlio_f1tenth::wall_points()) {
    types::Point<types::Point_XYZ> pt;
    pt.pos = p;
    wall.push_back(pt);
  }
  map->add_points(wall);

  types::RegistrationConfig config;
  types::RegistrationDebug debug;
  config.solver_type = "GaussNewton";
  config.max_iter = 100;
  config.convergence_criterion = 5.0e-3;
  config.damping_factor = 1.0e-9;
  config.damping_scale = 2.0;
  config.planar = true;
  auto registration = Reg<TConfig>::from_config(config, debug);
  registration->init();
  if constexpr (types::FRAMEMAP<TConfig>) registration->init_frame_map(map->get_config());

  farlio_f1tenth::Noise noise(20261026);
  std::vector<double> ms;
  const double lap = 2.0 * farlio_f1tenth::kStraight + 2.0 * farlio_f1tenth::kPi * farlio_f1tenth::kRadius;
  for (int i = 0; i < scans; ++i) {
    // one scan per 10 ms at 2 m/s: consecutive poses 2 cm apart, spread over the lap
    const auto pose = farlio_f1tenth::centerline_pose(std::fmod(i * lap / scans, lap));
    const Sophus::SE3f truth(
      Eigen::Quaternionf(Eigen::AngleAxisf(static_cast<float>(pose.yaw), Eigen::Vector3f::UnitZ())),
      Eigen::Vector3f(static_cast<float>(pose.x), static_cast<float>(pose.y), 0.0f));
    std::vector<types::Point<TConfig>> frame;
    for (const auto & h : farlio_f1tenth::cast_scan(pose, beams, &noise)) {
      types::Point<TConfig> pt{};
      pt.pos = h;
      frame.push_back(pt);
    }
    // guess = truth plus the 2 cm the car moved since the last scan
    const Sophus::SE3f guess = tam::core::state::utils::project_se2(truth * Sophus::SE3f(Eigen::Quaternionf::Identity(), Eigen::Vector3f(0.02f, 0.0f, 0.0f)));
    const auto t0 = std::chrono::steady_clock::now();
    registration->register_frame(frame, map.get(), guess, 1.0f, 0.1f);
    ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  }
  std::sort(ms.begin(), ms.end());
  const double mean = std::accumulate(ms.begin(), ms.end(), 0.0) / ms.size();
  const double p99 = ms[std::min<std::size_t>(ms.size() - 1, static_cast<std::size_t>(0.99 * ms.size()))];
  std::printf("FARLIO_1710 %-8s %4d beams: mean %.3f ms  p99 %.3f ms  max %.3f ms  (%d scans, CPU, 1 thread)\n", name,
    beams, mean, p99, ms.back(), scans);
}

int main(int argc, char * argv[])
{
  const int scans = argc > 1 ? std::atoi(argv[1]) : 200;
  for (const int beams : {64, 1080}) {
    bench<types::GICP_EXT, tam::core::state::GICP>("GICP_EXT", beams, scans);
    bench<types::ICP_EXT, tam::core::state::ICP>("ICP_EXT", beams, scans);
  }
  return 0;
}
