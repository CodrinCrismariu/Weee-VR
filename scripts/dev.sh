#!/usr/bin/env bash
# Build, install and launch WeeeVR on the connected iPhone. Run inside WSL:
#   scripts/dev.sh            build + install + launch
#   scripts/dev.sh build      build only
# The package is mirrored to the Linux filesystem first: SwiftPM builds over /mnt/c are very slow.
# Third-party C++ (XRSLAM, Ceres, Eigen, OpenCV) comes from scripts/fetch-xrslam.sh.
set -euo pipefail

SRC="$(cd "$(dirname "$0")/../native/WeeeVR" && pwd)"
DST="$HOME/weee-build/WeeeVR"
THIRD_PARTY="$HOME/weee-thirdparty/ThirdParty"
HAND_MODELS="$HOME/weee-thirdparty/HandModels"
if [ ! -d "$THIRD_PARTY" ]; then
    echo "Missing $THIRD_PARTY: run scripts/fetch-xrslam.sh first." >&2
    exit 1
fi
if [ ! -d "$HAND_MODELS/HandDetector.mlpackage" ]; then
    echo "Missing $HAND_MODELS: run ~/calib/venv/bin/python scripts/hands/fetch-models.py $HAND_MODELS" >&2
    exit 1
fi
mkdir -p "$DST"
rsync -a --delete --exclude .build --exclude .swiftpm --exclude xtool --exclude ThirdParty     --exclude Sources/WeeeVR/HandModels "$SRC/" "$DST/"
rsync -a --delete "$THIRD_PARTY/" "$DST/ThirdParty/"
rsync -a --delete --exclude "_*" "$HAND_MODELS/" "$DST/Sources/WeeeVR/HandModels/"

# shellcheck disable=SC1091
. "$HOME/.local/share/swiftly/env.sh"
cd "$DST"
exec xtool dev "$@"
