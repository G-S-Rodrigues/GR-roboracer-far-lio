#!/usr/bin/env bash
# Fork CI / local: CPU build, then FARLIO-1010..1030 against a golden recorded from upstream ef20510 in
# the SAME run (same machine, same flags), then every other CPU test. Runs inside the deps image.
#
#   /dev_ws/src/far_lio   the branch (mounted)
#   /ref_ws/src/far_lio   upstream ef20510 with the golden tests overlaid (mounted)
#
# The committed golden_3d_cpu.txt stays the default for runs without FARLIO_GOLDEN_PATH.
set -euo pipefail
. /opt/ros/jazzy/setup.bash
cd /dev_ws
colcon --log-base out/log build --build-base out/build-cpu --install-base out/install-cpu \
  --packages-up-to tam_odometry tam_state_estimation_node \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_CUDA=OFF -DBUILD_TESTING=ON

# Reference: upstream ef20510 (+ the golden test files), only tam_odometry, on top of the branch's deps.
. out/install-cpu/setup.bash
cd /ref_ws
colcon --log-base log build --build-base build --install-base install --packages-select tam_odometry \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_CUDA=OFF -DBUILD_TESTING=ON
export FARLIO_GOLDEN_PATH=/dev_ws/out/golden_ref_same_job.txt
FARLIO_RECORD_GOLDEN=1 build/tam_odometry/tam_odometry/odometry_pipeline/unit_3d_golden > /dev_ws/out/record.log 2>&1 || true
test -s "$FARLIO_GOLDEN_PATH" || { echo "reference golden was not recorded"; tail -20 /dev_ws/out/record.log; exit 1; }

cd /dev_ws
# frame-1 values at %.17g, branch vs same-job golden, for comparing a local run with a CI run
out/build-cpu/tam_odometry/tam_odometry/odometry_pipeline/unit_3d_golden 2>&1 | grep -a '^FRAME1' || true
colcon --log-base out/log test --build-base out/build-cpu --install-base out/install-cpu \
  --packages-select tam_odometry tam_state_estimation_node || true
colcon test-result --test-result-base out/build-cpu --verbose
