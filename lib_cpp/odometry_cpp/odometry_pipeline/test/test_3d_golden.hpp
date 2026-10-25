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
// NOTE: shared by unit_3d_golden.cpp (CPU) and unit_3d_golden.cu (CUDA). Define
// FARLIO_GOLDEN_FILE (absolute path) and FARLIO_GOLDEN_TOL (float) before including.
//
// FARLIO-1010..1040: the 3D odometry path, with every 2D feature off, is held to a golden recorded
// on unmodified upstream ef20510. Run with FARLIO_RECORD_GOLDEN=1 to (re)record. A recorded golden
// is a reviewed diff; never regenerate it to make a run pass.
#pragma once

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "odometry_pipeline/odometry_pipeline.hpp"
#include "param_management_cpp/param_reference_manager.hpp"
#include "tsl_logger_cpp/reference_logger.hpp"

namespace farlio_golden {
constexpr int kNumFrames = 25;

// Hand-rolled Box-Muller on a fixed mt19937: std::normal_distribution is implementation-defined
inline double gauss(std::mt19937& gen)
{
  const double u1 = (static_cast<double>(gen()) + 1.0) / 4294967297.0;
  const double u2 = (static_cast<double>(gen()) + 1.0) / 4294967297.0;
  return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
}

// Deterministic 3D scene in the world frame: ground, walls at mixed orientations, a tilted roof,
// and poles.
inline std::vector<Eigen::Vector3f> scene()
{
  std::vector<Eigen::Vector3f> pts;
  const float step = 0.3f;
  auto plane = [&](const Eigen::Vector3f& o, const Eigen::Vector3f& u, const Eigen::Vector3f& v, float lu, float lv) {
    for (float a = 0.0f; a < lu; a += step) {
      for (float b = 0.0f; b < lv; b += step) {
        pts.push_back(o + a * u + b * v);
      }
    }
  };
  // ground
  plane({-10.0f, -15.0f, -1.5f}, {1, 0, 0}, {0, 1, 0}, 60.0f, 30.0f);
  // wall ahead and behind
  plane({40.0f, -15.0f, -1.5f}, {0, 1, 0}, {0, 0, 1}, 30.0f, 6.0f);
  plane({-8.0f, -15.0f, -1.5f}, {0, 1, 0}, {0, 0, 1}, 30.0f, 6.0f);
  // side walls
  plane({-10.0f, 12.0f, -1.5f}, {1, 0, 0}, {0, 0, 1}, 50.0f, 6.0f);
  plane({-10.0f, -10.0f, -1.5f}, {0.9396926f, -0.3420201f, 0}, {0, 0, 1}, 30.0f, 6.0f);
  // tilted roof
  plane({5.0f, -8.0f, 4.0f}, {1, 0, 0}, {0, 0.9659258f, -0.2588190f}, 25.0f, 14.0f);
  // poles
  const float poles[5][2] = {{8.0f, 4.0f}, {14.0f, -5.0f}, {20.0f, 6.0f}, {26.0f, -3.0f}, {31.0f, 3.0f}};
  for (const auto& p : poles) {
    for (float z = -1.5f; z < 3.0f; z += 0.1f) {
      for (int k = 0; k < 12; ++k) {
        const float ang = k * 2.0f * static_cast<float>(M_PI) / 12.0f;
        pts.emplace_back(p[0] + 0.15f * std::cos(ang), p[1] + 0.15f * std::sin(ang), z);
      }
    }
  }
  return pts;
}

// Truth pose of frame k: motion in x, y, z, yaw, pitch and roll; frame 0 is the identity
inline Sophus::SE3f truth(int k)
{
  const double t = k;
  const Eigen::AngleAxisf yaw(static_cast<float>(0.02 * t), Eigen::Vector3f::UnitZ());
  const Eigen::AngleAxisf pitch(static_cast<float>(0.015 * std::sin(0.3 * t)), Eigen::Vector3f::UnitY());
  const Eigen::AngleAxisf roll(static_cast<float>(0.02 * std::sin(0.2 * t)), Eigen::Vector3f::UnitX());
  const Eigen::Quaternionf q = yaw * pitch * roll;
  return Sophus::SE3f(q, Eigen::Vector3f(static_cast<float>(0.12 * t), static_cast<float>(0.2 * std::sin(0.15 * t)),
                            static_cast<float>(0.03 * t)));
}

// Deterministic perturbation of the truth that stands in for the model's guess
inline Sophus::SE3f perturbed(int k)
{
  const double s = 1.0 + 0.3 * std::sin(1.0 * k);
  const Sophus::SE3f d(
    Eigen::Quaternionf(Eigen::AngleAxisf(static_cast<float>(0.005 * s), Eigen::Vector3f::UnitZ())),
    Eigen::Vector3f(static_cast<float>(0.03 * s), static_cast<float>(-0.02 * s), static_cast<float>(0.015 * s)));
  return truth(k) * d;
}

template <typename TConfig>
std::vector<tam::core::state::types::Point<TConfig>> frame_at(const std::vector<Eigen::Vector3f>& world, int k)
{
  std::mt19937 gen(1000u + static_cast<unsigned>(k));
  const Sophus::SE3f inv = truth(k).inverse();
  const Eigen::Vector3f origin = truth(k).translation();
  std::vector<tam::core::state::types::Point<TConfig>> frame;
  for (const auto& p : world) {
    if ((p - origin).norm() > 35.0f) continue;
    // per-frame subset: different points every frame, so registration is not trivially exact
    if ((gen() & 3u) != 0u) continue;
    tam::core::state::types::Point<TConfig> pt{};
    const Eigen::Vector3f noise(static_cast<float>(0.01 * gauss(gen)), static_cast<float>(0.01 * gauss(gen)),
      static_cast<float>(0.01 * gauss(gen)));
    pt.pos = inv * p + noise;
    frame.push_back(pt);
  }
  return frame;
}

inline std::string pose_line(const std::string& name, int k, const Sophus::SE3f& p)
{
  std::ostringstream os;
  os << std::hexfloat << name << " " << k << " " << p.translation().x() << " " << p.translation().y() << " "
     << p.translation().z() << " " << p.unit_quaternion().w() << " " << p.unit_quaternion().x() << " "
     << p.unit_quaternion().y() << " " << p.unit_quaternion().z();
  return os.str();
}

struct GoldenPose {
  float v[7];
};

inline std::map<std::pair<std::string, int>, GoldenPose> read_golden(const std::string& path)
{
  std::map<std::pair<std::string, int>, GoldenPose> out;
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream is(line);
    std::string name, tok;
    int k = 0;
    is >> name >> k;
    GoldenPose g{};
    for (int i = 0; i < 7; ++i) {
      is >> tok;
      g.v[i] = std::strtof(tok.c_str(), nullptr);
    }
    out[{name, k}] = g;
  }
  return out;
}

// Appends (or, when recording, starts) the golden lines for one case
inline void record(const std::string& path, const std::vector<std::string>& lines)
{
  static bool first = true;  // the first case of a recording run truncates the file
  std::ofstream out(path, first ? std::ios::trunc : std::ios::app);
  if (first) out << "# FARLIO golden: name frame x y z qw qx qy qz (hexfloat). See test_3d_golden.hpp\n";
  first = false;
  for (const auto& l : lines) out << l << "\n";
}

template <typename TConfig>
void run_case(const std::string& name, const std::string& solver)
{
  const auto world = scene();
  tam::pmg::ParamReferenceManager::UniquePtr pmg = std::make_unique<tam::pmg::ParamReferenceManager>();
  tam::tsl::ReferenceLogger::UniquePtr logger = std::make_unique<tam::tsl::ReferenceLogger>();
  auto pipeline = tam::core::state::OdometryPipeline<TConfig>::from_config(pmg.get(), logger.get());
  tam::pmg::MgmtInterface* p = pmg.get();
  p->set_value("pipeline.update_map", true);
  p->set_value("pipeline.undistort", false);
  p->set_value("pipeline.preprocess", false);
  p->set_value("pipeline.downsample", false);
  p->set_value("pipeline.buffer_z", false);
  p->set_value("pipeline.num_threads", std::int64_t{1});
  p->set_value("registration.max_time", 1.0e9);
  p->set_value("registration.max_iter", std::int64_t{50});
  p->set_value("registration.convergence_criterion", 5.0e-3);
  p->set_value("registration.damping_factor", 1.0e-9);
  p->set_value("registration.solver_type", solver);
  p->set_value("map.voxel_size", 1.0);
  p->set_value("map.cov_regularization", std::string("SVD"));
  // fixed threshold: the model deviation never updates it
  p->set_value("threshold.initial_threshold", 1.0);
  p->set_value("threshold.min_motion_threshold", 1.0e6);
  pipeline->init();

  const char* rec_env = std::getenv("FARLIO_RECORD_GOLDEN");
  const bool recording = rec_env != nullptr && std::string(rec_env) == "1";
  std::vector<std::string> lines;
  std::vector<Sophus::SE3f> poses;
  std::vector<Sophus::SE3f> guesses;

  for (int k = 0; k < kNumFrames; ++k) {
    const std::uint64_t stamp = 1000000000ull + static_cast<std::uint64_t>(k) * 100000000ull;
    // frame 0 seeds the map at the identity; later frames start from a perturbed truth
    // (the constant-velocity model predicts from its own history; only frame 0 is seeded for it)
    Sophus::SE3f guess = (k == 0) ? truth(0) : perturbed(k);
    if constexpr (TConfig::MODEL == tam::core::state::types::ModelType::EXTERNALGUESS) {
      tam::core::state::types::PoseStamped ps{};
      ps.stamp = stamp;
      ps.pose = guess;
      pipeline->set_pose_model(ps, true);
    } else if (k == 0) {
      tam::core::state::types::PoseStamped ps{};
      ps.stamp = stamp;
      ps.pose = guess;
      pipeline->set_pose_model(ps, true);
    } else {
      guess = Sophus::SE3f();  // not an externally supplied guess: skip the "moved off the guess" check
    }
    const auto frame = frame_at<TConfig>(world, k);
    const auto odom = pipeline->register_frame(frame, stamp);
    poses.push_back(odom.pose.pose);
    guesses.push_back(guess);
    lines.push_back(pose_line(name, k, odom.pose.pose));
  }

  // The golden cannot be garbage: close to truth, and moved off the guess after frame 0
  // ICP_CV predicts from its own history and starts from the identity with zero velocity. Measured
  // on upstream ef20510 it tracks the truth only to about 0.3 m / 2 deg here, so its bound is
  // looser than the externally guided configs' (5 cm / 0.5 deg); the golden is what pins it.
  constexpr bool kGuided = TConfig::MODEL == tam::core::state::types::ModelType::EXTERNALGUESS;
  const float max_t = kGuided ? 0.05f : 0.5f;
  const float max_r = (kGuided ? 0.5f : 3.0f) * static_cast<float>(M_PI) / 180.0f;
  for (int k = 0; k < kNumFrames; ++k) {
    const Sophus::SE3f err = truth(k).inverse() * poses[k];
    EXPECT_LT(err.translation().norm(), max_t) << name << " frame " << k;
    EXPECT_LT(err.so3().log().norm(), max_r) << name << " frame " << k;
    if (k > 0 && TConfig::MODEL == tam::core::state::types::ModelType::EXTERNALGUESS) {
      const Sophus::SE3f moved = guesses[k].inverse() * poses[k];
      EXPECT_GT(moved.translation().norm() + moved.so3().log().norm(), 1.0e-6f) << name << " frame " << k;
    }
  }

  const std::string path = FARLIO_GOLDEN_FILE;
  if (recording) {
    record(path, lines);
    return;
  }
  std::ifstream probe(path);
  ASSERT_TRUE(probe.good()) << "missing golden " << path << " (a missing golden is a failure, not a skip)";
  const auto golden = read_golden(path);
  for (int k = 0; k < kNumFrames; ++k) {
    const auto it = golden.find({name, k});
    ASSERT_NE(it, golden.end()) << "no golden for " << name << " frame " << k;
    const auto& g = it->second;
    const Sophus::SE3f& r = poses[k];
    const float got[7] = {r.translation().x(), r.translation().y(), r.translation().z(), r.unit_quaternion().w(),
      r.unit_quaternion().x(), r.unit_quaternion().y(), r.unit_quaternion().z()};
    for (int i = 0; i < 7; ++i) {
      if (FARLIO_GOLDEN_TOL == 0.0f) {
        EXPECT_EQ(got[i], g.v[i]) << name << " frame " << k << " component " << i;
      } else {
        EXPECT_NEAR(got[i], g.v[i], FARLIO_GOLDEN_TOL) << name << " frame " << k << " component " << i;
      }
    }
  }
}
}  // namespace farlio_golden
