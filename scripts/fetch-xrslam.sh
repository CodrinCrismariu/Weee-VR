#!/usr/bin/env bash
# Fetch XRSLAM and its dependencies at pinned versions and assemble the C++ sources that the Swift
# package compiles (Package.swift targets CeresSolver and XRSLAM). Run inside WSL:
#   scripts/fetch-xrslam.sh
# Output goes to the Linux filesystem (~/weee-thirdparty/ThirdParty, ~600 MB with OpenCV), not into
# the repo; scripts/dev.sh mirrors it next to the package. Our own changes live in the repo:
#   native/thirdparty/overlay/   hand-written headers that CMake would normally generate
#   native/thirdparty/patches/   minimal diffs against upstream XRSLAM
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
HERE="$REPO/native/thirdparty"
CACHE="$HOME/weee-thirdparty/cache"
OUT="$HOME/weee-thirdparty/ThirdParty"

XRSLAM_REV=4beb1a942f33da9afbfae2d70e2c641cfc2bb675   # openxrlab/xrslam, Feb 2026
CERES_REV=e809cf0c2879f521078b4c9e6329390b42ecf722    # Ceres 1.14 + std::function fix (XRSLAM's pin)
EIGEN_TAG=3.3.7
OPENCV_VERSION=4.10.0

mkdir -p "$CACHE"

# git_at <dir> <url> <commit-or-tag>: shallow checkout of one revision, reset to pristine.
git_at() {
    local dir="$CACHE/$1"
    if [ ! -d "$dir/.git" ]; then
        git init -q "$dir"
        git -C "$dir" remote add origin "$2"
    fi
    if ! git -C "$dir" rev-parse -q --verify "$3^{commit}" >/dev/null; then
        git -C "$dir" fetch -q --depth 1 origin "$3"
    fi
    git -C "$dir" -c advice.detachedHead=false checkout -q -f "$3" 2>/dev/null \
        || git -C "$dir" -c advice.detachedHead=false checkout -q -f FETCH_HEAD
    git -C "$dir" clean -q -fdx
}

echo "== sources"
git_at xrslam https://github.com/openxrlab/xrslam.git "$XRSLAM_REV"
git_at ceres https://github.com/ceres-solver/ceres-solver.git "$CERES_REV"
git_at eigen https://github.com/eigenteam/eigen-git-mirror.git "$EIGEN_TAG"

echo "== patches"
for p in "$HERE"/patches/*.diff; do
    name="$(basename "$p")"
    echo "   $name"
    patch -s -p1 -d "$CACHE/${name%%-*}" < "$p"     # xrslam-*.diff -> cache/xrslam, ceres-*.diff -> cache/ceres
done

echo "== OpenCV $OPENCV_VERSION (prebuilt iOS framework, arm64 slice)"
ZIP="$CACHE/opencv-$OPENCV_VERSION-ios-framework.zip"
[ -f "$ZIP" ] || curl -fL --progress-bar -o "$ZIP" \
    "https://github.com/opencv/opencv/releases/download/$OPENCV_VERSION/opencv-$OPENCV_VERSION-ios-framework.zip"
rm -rf "$CACHE/opencv-unzip" && mkdir -p "$CACHE/opencv-unzip"
unzip -q "$ZIP" -d "$CACHE/opencv-unzip"
FW="$CACHE/opencv-unzip/opencv2.framework/Versions/A"

rm -rf "$OUT"
mkdir -p "$OUT/opencv/include/opencv2" "$OUT/opencv/lib"
# C++ headers only (the framework also carries Objective-C/Swift wrappers we don't use).
for d in "$FW"/Headers/*/; do cp -r "$d" "$OUT/opencv/include/opencv2/"; done
cp "$FW"/Headers/*.hpp "$OUT/opencv/include/opencv2/"
python3 "$REPO/scripts/macho-slice.py" "$FW/opencv2" arm64 "$OUT/opencv/lib/libopencv2.a"
rm -rf "$CACHE/opencv-unzip"

echo "== Eigen $EIGEN_TAG"
mkdir -p "$OUT/eigen"
cp -r "$CACHE/eigen/Eigen" "$CACHE/eigen/unsupported" "$OUT/eigen/"   # OpenCV's core/eigen.hpp wants Tensor

echo "== Ceres (Eigen sparse, miniglog, dynamic Schur only)"
C="$CACHE/ceres"
mkdir -p "$OUT/ceres/internal/ceres/generated" "$OUT/ceres/internal/ceres/miniglog/glog"
cp -r "$C/include" "$OUT/ceres/"
cp -r "$HERE/overlay/ceres/config" "$OUT/ceres/"
cp "$C"/internal/ceres/*.h "$OUT/ceres/internal/ceres/"
# The library's own source list (internal/ceres/CMakeLists.txt), not the tests and benchmarks.
sed -n '/^set(CERES_INTERNAL_SRC/,/^)/p' "$C/internal/ceres/CMakeLists.txt" | grep -o '[a-z_0-9]*\.cc' \
    | while read -r f; do cp "$C/internal/ceres/$f" "$OUT/ceres/internal/ceres/"; done
cp "$C"/internal/ceres/generated/*_d_d_d.cc "$OUT/ceres/internal/ceres/generated/"
cp "$C"/internal/ceres/miniglog/glog/* "$OUT/ceres/internal/ceres/miniglog/glog/"

echo "== XRSLAM core + OpenCV image"
X="$CACHE/xrslam"
mkdir -p "$OUT/xrslam/include/xrslam" "$OUT/xrslam/extra/src" "$OUT/xrslam/extra/include"
cp -r "$X/xrslam/src" "$OUT/xrslam/"
cp "$X"/xrslam/include/xrslam/*.h "$OUT/xrslam/include/xrslam/"
cp "$HERE/overlay/xrslam/include/xrslam/version.h" "$OUT/xrslam/include/xrslam/"
# Logging goes to os_log from our wrapper (Sources/XRSLAMCore/XRSLAMLog.cpp) instead of spdlog.
rm "$OUT/xrslam/src/xrslam/utility/debug.cpp"
cp -r "$X/xrslam-extra/include/xrslam" "$OUT/xrslam/extra/include/"
cp "$X/xrslam-extra/src/xrslam/extra/opencv_image.cpp" "$OUT/xrslam/extra/src/"
rm "$OUT/xrslam/extra/include/xrslam/extra/yaml_config.h"   # needs yaml-cpp; we configure from code

cat > "$OUT/VERSIONS.txt" <<EOF
xrslam   $XRSLAM_REV  (Apache-2.0)
ceres    $CERES_REV   (BSD-3-Clause)
eigen    $EIGEN_TAG   (MPL-2.0; Ceres' Eigen sparse backend uses Eigen's LGPL AMD ordering)
opencv   $OPENCV_VERSION (Apache-2.0)
EOF
echo "== done: $OUT ($(du -sh "$OUT" | cut -f1))"
