#!/usr/bin/env bash
# Prepare the reference tree for scripts/ci_cpu.sh: upstream ef20510 plus the golden test files.
# usage: ci_prepare_ref.sh <dest-dir>   (run from the branch checkout; needs full history)
set -euo pipefail
dest=$1
rev=ef20510316ad1f8dbf2e428e4a193f700493eccd
git cat-file -e "$rev^{commit}" || { echo "upstream $rev is not in this clone (fetch-depth: 0)"; exit 1; }
rm -rf "$dest"
git worktree add --detach "$dest" "$rev"
# same TAM__common pin on both sides, so reuse the checked-out submodules
test "$(git ls-tree "$rev" submodules/TAM__common | awk '{print $3}')" = \
  "$(git ls-tree HEAD submodules/TAM__common | awk '{print $3}')"
rm -rf "$dest/submodules"
cp -a submodules "$dest/submodules"
od=lib_cpp/odometry_cpp/odometry_pipeline
cp "$od/CMakeLists.txt" "$dest/$od/CMakeLists.txt"
# only the golden test files go in: upstream's own tests stay upstream's (the branch's other tests use the
# branch's API and do not compile against ef20510)
for f in test_3d_golden.hpp unit_3d_golden.cpp unit_3d_golden.cu; do
  cp "$od/test/$f" "$dest/$od/test/$f"
done
test ! -d "$od/test/data" || cp -a "$od/test/data" "$dest/$od/test/"
