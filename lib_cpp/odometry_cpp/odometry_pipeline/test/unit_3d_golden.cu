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
// FARLIO-1040: the CUDA twin of FARLIO-1010 / 1020, against its own golden (golden_3d_cuda.txt).
// Device atomics reorder reduction terms, so the comparison is within FARLIO_GOLDEN_TOL, a measured
// spread (see the comment beside the constant). Skips when no CUDA device is present (CI compiles
// it and skips); a missing golden file is a failure, not a skip.
#include <cuda_runtime.h>

#define FARLIO_GOLDEN_FILE FARLIO_GOLDEN_DIR "/golden_3d_cuda.txt"
// Tolerance = 2 x the measured max per-component spread. Measured 2026-10-26 on unmodified-algorithm
// ef20510: N = 6 recordings, NVIDIA GeForce RTX 4060 Laptop GPU (sm_89), CUDA 12.8.1, Release;
// all six golden files byte-identical (md5 d83b6d88...), so spread = 0 and tolerance = 0.
#define FARLIO_GOLDEN_TOL 0.0f
#include "test_3d_golden.hpp"

using namespace tam::core::state::types;  // NOLINT

#define FARLIO_REQUIRE_CUDA_DEVICE()                         \
  do {                                                       \
    int count = 0;                                           \
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) { \
      GTEST_SKIP() << "no CUDA device";                      \
    }                                                        \
  } while (0)

TEST(FARLIO_1040, CUDA_GICP_EXT_GaussNewton)
{
  FARLIO_REQUIRE_CUDA_DEVICE();
  farlio_golden::run_case<CUDA_GICP_EXT>("cuda_gicp_ext_gn", "GaussNewton");
}
TEST(FARLIO_1040, CUDA_GICP_EXT_LevenbergMarquardt)
{
  FARLIO_REQUIRE_CUDA_DEVICE();
  farlio_golden::run_case<CUDA_GICP_EXT>("cuda_gicp_ext_lm", "LevenbergMarquardt");
}
TEST(FARLIO_1040, CUDA_ICP_EXT_GaussNewton)
{
  FARLIO_REQUIRE_CUDA_DEVICE();
  farlio_golden::run_case<CUDA_ICP_EXT>("cuda_icp_ext_gn", "GaussNewton");
}
TEST(FARLIO_1040, CUDA_ICP_EXT_LevenbergMarquardt)
{
  FARLIO_REQUIRE_CUDA_DEVICE();
  farlio_golden::run_case<CUDA_ICP_EXT>("cuda_icp_ext_lm", "LevenbergMarquardt");
}
