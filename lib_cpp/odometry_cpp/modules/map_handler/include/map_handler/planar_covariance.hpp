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

#include <Eigen/Dense>
#include <cmath>

#if defined(__CUDACC__)
#define TAM_HD __host__ __device__
#else
#define TAM_HD
#endif

namespace tam::core::state::utils
{
/**
 * @brief Make a neighbourhood covariance planar (analytic vertical extrusion)
 *
 * A single-layer scan's neighbourhood has two near-zero eigenvalues (across the wall, and z), so the
 * smallest eigenvector, which becomes the normal, can be z and constrain nothing in the plane. The z
 * row and column are replaced by Szz = max(1, trace) with zero cross terms, so the smallest eigenvector
 * is always in the plane.
 */
TAM_HD inline void planarize_covariance(Eigen::Matrix3f & c)
{
  const float trace = c(0, 0) + c(1, 1) + c(2, 2);
  c(0, 2) = 0.0f;
  c(1, 2) = 0.0f;
  c(2, 0) = 0.0f;
  c(2, 1) = 0.0f;
  c(2, 2) = fmaxf(1.0f, trace);
}
}  // namespace tam::core::state::utils
