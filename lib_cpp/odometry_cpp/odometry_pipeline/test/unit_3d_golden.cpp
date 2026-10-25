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
// FARLIO-1010 (GICP_EXT), 1020 (ICP_EXT), 1030 (ICP_CV): CPU, bit-exact.
#define FARLIO_GOLDEN_FILE FARLIO_GOLDEN_DIR "/golden_3d_cpu.txt"
#define FARLIO_GOLDEN_TOL 0.0f
#include "test_3d_golden.hpp"

using namespace tam::core::state::types;  // NOLINT

TEST(FARLIO_1010, GICP_EXT_GaussNewton) { farlio_golden::run_case<GICP_EXT>("gicp_ext_gn", "GaussNewton"); }
TEST(FARLIO_1010, GICP_EXT_LevenbergMarquardt)
{
  farlio_golden::run_case<GICP_EXT>("gicp_ext_lm", "LevenbergMarquardt");
}
TEST(FARLIO_1020, ICP_EXT_GaussNewton) { farlio_golden::run_case<ICP_EXT>("icp_ext_gn", "GaussNewton"); }
TEST(FARLIO_1020, ICP_EXT_LevenbergMarquardt)
{
  farlio_golden::run_case<ICP_EXT>("icp_ext_lm", "LevenbergMarquardt");
}
TEST(FARLIO_1030, ICP_CV_GaussNewton) { farlio_golden::run_case<ICP_CV>("icp_cv_gn", "GaussNewton"); }
TEST(FARLIO_1030, ICP_CV_LevenbergMarquardt)
{
  farlio_golden::run_case<ICP_CV>("icp_cv_lm", "LevenbergMarquardt");
}
