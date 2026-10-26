#!/usr/bin/env bash
# Fork CI / local: CPU build, then FARLIO-1010..1030 against a golden recorded from upstream ef20510 in
# the SAME run (same machine, same flags), then every other CPU test. Runs inside the deps image.
#
#   /dev_ws/src/far_lio   the branch (mounted)
#   /ref_ws/src/far_lio   upstream ef20510 with the golden tests overlaid (mounted)
#
# The committed golden_3d_cpu.txt stays the default for runs without FARLIO_GOLDEN_PATH.
set -euo pipefail
set +u  # ROS setup scripts read unset variables
mkdir -p /dev_ws/out  # CI mounts no volume here; the local run mounts one

# Reference: upstream ef20510 (+ the golden test files), built in its own workspace, with its own deps and
# its own install prefix, in a subshell that has sourced nothing but ROS. The pipeline is header-only, so a
# reference compiled against the branch's installed headers would be the branch (a 1e-5 nudge moved the
# "golden" by exactly 1e-5 in run 40269706806).
(
  . /opt/ros/jazzy/setup.bash
  cd /ref_ws
  colcon --log-base log build --build-base build --install-base install --packages-up-to tam_odometry \
    --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_CUDA=OFF -DBUILD_TESTING=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
  # the reference's include path, from its compilation database: nothing from the branch
  echo "reference include path (unit_3d_golden):"
  python3 - <<'PY'
import json, shlex, sys
for e in json.load(open("/ref_ws/build/tam_odometry/compile_commands.json")):
    if e["file"].endswith("unit_3d_golden.cpp"):
        inc = [t for t in shlex.split(e["command"]) if t.startswith(("-I", "-isystem")) or t.startswith("/")]
        print("\n".join(inc))
        bad = [t for t in inc if "/dev_ws" in t]
        sys.exit("reference sees the branch: %s" % bad if bad else 0)
PY
  . install/setup.bash
  export FARLIO_GOLDEN_PATH=/dev_ws/out/golden_ref_same_job.txt
  FARLIO_RECORD_GOLDEN=1 build/tam_odometry/tam_odometry/odometry_pipeline/unit_3d_golden > /dev_ws/out/record.log 2>&1 || true
  test -s "$FARLIO_GOLDEN_PATH" || { echo "reference golden was not recorded"; tail -20 /dev_ws/out/record.log; exit 1; }
  echo "upstream ef20510, 3D errors against truth (ERR3D):"
  build/tam_odometry/tam_odometry/odometry_pipeline/unit_3d_golden 2>&1 | grep -a '^ERR3D' || true
)

. /opt/ros/jazzy/setup.bash
cd /dev_ws
colcon --log-base out/log build --build-base out/build-cpu --install-base out/install-cpu \
  --packages-up-to tam_odometry tam_state_estimation_node \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_CUDA=OFF -DBUILD_TESTING=ON
. out/install-cpu/setup.bash
set -u
export FARLIO_GOLDEN_PATH=/dev_ws/out/golden_ref_same_job.txt

# frame-1 values at %.17g and the errors against truth, for comparing a local run with a CI run
echo "branch:"
out/build-cpu/tam_odometry/tam_odometry/odometry_pipeline/unit_3d_golden 2>&1 | grep -a '^FRAME1\|^ERR3D' || true
colcon --log-base out/log test --build-base out/build-cpu --install-base out/install-cpu \
  --packages-select tam_odometry tam_state_estimation_node || true
colcon test-result --test-result-base out/build-cpu --verbose
