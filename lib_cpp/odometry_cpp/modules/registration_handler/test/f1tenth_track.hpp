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
// Test-only fixture (FARLIO-1100 and later): a synthetic F1TENTH-scale track and a deterministic 2D
// ray-caster. Fork-side; it is not part of the library.
//
// The track is a stadium oval: two straights of 8 m joined by half circles of centerline radius
// 3 m, half-width 1.1 m, so the walls are stadiums of radius 1.9 m (inner) and 4.1 m (outer).
// Beams sit at their declared angles (gotcha: never index-compare scans), -fov/2 + i*fov/(N-1).
// Noise comes from a fixed std::mt19937 and a hand-rolled Box-Muller: std::normal_distribution is
// implementation-defined, so a golden built on it would not be portable.

#include <Eigen/Dense>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace farlio_f1tenth
{
constexpr double kStraight = 8.0;      // m, length of each straight
constexpr double kRadius = 3.0;        // m, centerline radius of the end arcs
constexpr double kHalfWidth = 1.1;     // m
constexpr double kFov = 4.7;           // rad
constexpr double kRangeMax = 10.0;     // m
constexpr double kNoiseSigma = 0.03;   // m
constexpr double kWallSpacing = 0.05;  // m, spacing of sampled wall points (survey grid cell)
constexpr double kPi = 3.14159265358979323846;

/// Closed polyline of the stadium wall at offset `offset` from the centerline (counter-clockwise).
inline std::vector<Eigen::Vector2d> stadium_wall(const double offset, const double spacing)
{
  const double r = kRadius + offset;
  const double half = kStraight / 2.0;
  std::vector<Eigen::Vector2d> pts;
  // bottom straight, left to right (y = -r)
  for (double x = -half; x < half; x += spacing) pts.emplace_back(x, -r);
  // right arc, centre (half, 0), from -90 deg to +90 deg
  const double da = spacing / r;
  for (double a = -kPi / 2.0; a < kPi / 2.0; a += da)
    pts.emplace_back(half + r * std::cos(a), r * std::sin(a));
  // top straight, right to left (y = +r)
  for (double x = half; x > -half; x -= spacing) pts.emplace_back(x, r);
  // left arc, centre (-half, 0), from +90 deg to +270 deg
  for (double a = kPi / 2.0; a < 3.0 * kPi / 2.0; a += da)
    pts.emplace_back(-half + r * std::cos(a), r * std::sin(a));
  return pts;
}

/// Wall points sampled every kWallSpacing, both walls, z = 0 (what a survey-grid map holds).
inline std::vector<Eigen::Vector3f> wall_points()
{
  std::vector<Eigen::Vector3f> out;
  for (const double offset : {-kHalfWidth, kHalfWidth})
    for (const auto & p : stadium_wall(offset, kWallSpacing))
      out.emplace_back(static_cast<float>(p.x()), static_cast<float>(p.y()), 0.0f);
  return out;
}

/// Pose on the centerline at arc length s along the lap (counter-clockwise from (0, -3)).
struct Pose2D {
  double x, y, yaw;
};

inline Pose2D centerline_pose(const double s)
{
  const double half = kStraight / 2.0;
  const double arc = kPi * kRadius;
  const double lap = 2.0 * kStraight + 2.0 * arc;
  double u = std::fmod(s, lap);
  if (u < 0.0) u += lap;
  if (u < kStraight) return {-half + u, -kRadius, 0.0};
  u -= kStraight;
  if (u < arc) {
    const double a = -kPi / 2.0 + u / kRadius;
    return {half + kRadius * std::cos(a), kRadius * std::sin(a), a + kPi / 2.0};
  }
  u -= arc;
  if (u < kStraight) return {half - u, kRadius, kPi};
  u -= kStraight;
  const double a = kPi / 2.0 + u / kRadius;
  return {-half + kRadius * std::cos(a), kRadius * std::sin(a), a + kPi / 2.0};
}

/// Deterministic noise source: fixed mt19937, hand-rolled Box-Muller.
class Noise
{
public:
  explicit Noise(const std::uint32_t seed) : gen_(seed) {}
  double gaussian()
  {
    const double u1 = (static_cast<double>(gen_()) + 0.5) / 4294967296.0;
    const double u2 = (static_cast<double>(gen_()) + 0.5) / 4294967296.0;
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * kPi * u2);
  }

private:
  std::mt19937 gen_;
};

/// Ray-cast `num_beams` beams from `pose`; returns hit points in the sensor frame (z = 0).
/// Beams with no hit inside kRangeMax are dropped. `noise` may be null for a noiseless scan.
inline std::vector<Eigen::Vector3f> cast_scan(
  const Pose2D & pose, const int num_beams, Noise * noise = nullptr)
{
  // 0.01 m wall segments are fine enough that the polyline error is far below the 3 cm noise
  std::vector<std::vector<Eigen::Vector2d>> walls;
  for (const double offset : {-kHalfWidth, kHalfWidth}) walls.push_back(stadium_wall(offset, 0.01));
  std::vector<Eigen::Vector3f> hits;
  hits.reserve(num_beams);
  for (int i = 0; i < num_beams; ++i) {
    const double ang = -kFov / 2.0 + i * kFov / (num_beams - 1);
    const double world = pose.yaw + ang;
    const Eigen::Vector2d o(pose.x, pose.y);
    const Eigen::Vector2d d(std::cos(world), std::sin(world));
    double best = std::numeric_limits<double>::infinity();
    for (const auto & wall : walls) {
      for (std::size_t k = 0; k < wall.size(); ++k) {
        const Eigen::Vector2d a = wall[k];
        const Eigen::Vector2d e = wall[(k + 1) % wall.size()] - a;
        const double den = d.x() * e.y() - d.y() * e.x();
        if (std::abs(den) < 1e-12) continue;
        const Eigen::Vector2d ao = a - o;
        const double t = (ao.x() * e.y() - ao.y() * e.x()) / den;
        const double u = (ao.x() * d.y() - ao.y() * d.x()) / den;
        if (t > 0.0 && u >= 0.0 && u <= 1.0 && t < best) best = t;
      }
    }
    if (best > kRangeMax) continue;
    const double range = best + (noise ? kNoiseSigma * noise->gaussian() : 0.0);
    hits.emplace_back(
      static_cast<float>(range * std::cos(ang)), static_cast<float>(range * std::sin(ang)), 0.0f);
  }
  return hits;
}
}  // namespace farlio_f1tenth
