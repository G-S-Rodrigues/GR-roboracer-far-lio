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
#include <string>
#include <vector>

#include "test_gicp_parameterized.hpp"

/**
 * @brief Test construction of CudaGICP from param manager and logger
 */
TEST(CUDAGICP, BuildPmgLogger)
{
  // Check if CUDA device is available
  CHECK_CUDA_ERROR();
  // Initialize ICP
  // clang-format off
  // Construct from param manager and logger
  tam::pmg::ParamReferenceManager::UniquePtr pmg_ =  std::make_unique<tam::pmg::ParamReferenceManager>(); // NOLINT
  tam::tsl::ReferenceLogger::UniquePtr logger_ = std::make_unique<tam::tsl::ReferenceLogger>(); // NOLINT
  auto registration_ = tam::core::state::cuda::GICP<tam::core::state::types::CUDA_GICP_EXT>::from_config(pmg_.get(), logger_.get()); // NOLINT
  // clang-format on

  tam::pmg::MgmtInterface* pmg_raw = pmg_.get();
  pmg_raw->set_value("registration.max_iter", std::int64_t{100});

  EXPECT_EQ(registration_->get_config().max_iter, 100) << "Failed to construct ICP from param manager and logger";
}

/**
 * @brief Test construction of CudaGICP from config and debug object
 */
TEST(CUDAGICP, BuildConfigDebug)
{
  // Check if CUDA device is available
  CHECK_CUDA_ERROR();
  // clang-format off
  // Construct from config and debug objects
  tam::core::state::types::RegistrationConfig config;
  tam::core::state::types::RegistrationDebug debug;
  config.max_iter = 500;
  config.max_time = 150.0;
  config.convergence_criterion = 5e-3;
  auto registration_ = tam::core::state::cuda::GICP<tam::core::state::types::CUDA_GICP_EXT>::from_config(config, debug); // NOLINT
  // clang-format on
  EXPECT_EQ(registration_->get_config().max_iter, 500) << "Failed to construct ICP from config and debug objects";
}

/**
 * @brief Instantiate tests with various parameter combinations
 */
INSTANTIATE_TEST_SUITE_P(CUDAGICPCombinations, GICPParameterizedTest,
  ::testing::Values(TestParams{"SVD", "GaussNewton"}, TestParams{"SVD", "LevenbergMarquardt"},
    TestParams{"MIN_EIGENVALUE", "GaussNewton"}, TestParams{"MIN_EIGENVALUE", "LevenbergMarquardt"},
    TestParams{"FROBENIUS", "GaussNewton"}, TestParams{"FROBENIUS", "LevenbergMarquardt"}),
  [](const ::testing::TestParamInfo<TestParams>& info) { return info.param.string(); });

#include "test_planar_parameterized.hpp"
// One case per process under ctest (gtest_discover_tests runs each TEST_P instance separately). On this GPU
// (RTX 4060, 8 GiB) every cuda::GICP case in a process keeps about 1.35 GiB after its map and matcher are
// destroyed (measured: used memory 2450 / 3804 / 5158 / 6512 / 7866 MiB after cases 1..5, unchanged by
// cudaDeviceSynchronize and cudaMemPoolTrimTo; cuda::ICP does not do this), so six cases in one process end in
// cudaErrorMemoryAllocation. One instance per (row, solver) keeps each process at one case.
class Farlio1210GicpCuda : public ::testing::TestWithParam<int>
{
};
TEST_P(Farlio1210GicpCuda, PlanarSolve)
{
  CHECK_CUDA_ERROR();
  const int i = GetParam();
  check_1210_row<tam::core::state::types::CUDA_GICP_EXT, tam::core::state::cuda::GICP>(
    kGicp1210Cuda[i / 2], i % 2 == 0 ? "GaussNewton" : "LevenbergMarquardt");
}
INSTANTIATE_TEST_SUITE_P(FARLIO_1210_CUDA_GICP, Farlio1210GicpCuda, ::testing::Range(0, 20),
  [](const ::testing::TestParamInfo<int>& info) {
    const auto& b = kGicp1210Cuda[info.param / 2];
    return "s" + std::to_string(static_cast<int>(b.s)) + "_" + std::to_string(b.beams) +
           (info.param % 2 == 0 ? "_GN" : "_LM");
  });

class Farlio1220GicpCuda : public ::testing::TestWithParam<int>
{
};
TEST_P(Farlio1220GicpCuda, PlanarKeepsZRollPitch)
{
  CHECK_CUDA_ERROR();
  const int i = GetParam();
  check_1220_1230<tam::core::state::types::CUDA_GICP_EXT, tam::core::state::cuda::GICP>(
    i / 2 == 0 ? "GaussNewton" : "LevenbergMarquardt", i % 2 == 0 ? 0.0f : 1.0f);
}
INSTANTIATE_TEST_SUITE_P(FARLIO_1220_CUDA_GICP, Farlio1220GicpCuda, ::testing::Range(0, 4));

TEST(FARLIO_1240, CUDA_GICP_NonPlanarDrifts)
{
  CHECK_CUDA_ERROR();
  run_1240<tam::core::state::types::CUDA_GICP_EXT, tam::core::state::cuda::GICP>();
}
