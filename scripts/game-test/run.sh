#!/usr/bin/env bash
# Build and run the game core's self-test on Linux (WSL). The game sources use only Foundation and
# the standard library's SIMD types, so they compile here as they are; GameAudio (AVFoundation) is
# left out.
set -e
source ~/.local/share/swiftly/env.sh
HERE="$(cd "$(dirname "$0")" && pwd)"
GAME="$HERE/../../native/WeeeVR/Sources/WeeeVR/Game"
OUT="${TMPDIR:-$HOME}/weee-game-test"
mkdir -p "$OUT"
SOURCES=()
for f in "$GAME"/*.swift; do [[ "$f" == *GameAudio* ]] || SOURCES+=("$f"); done
swiftc -O -module-name GameTest -o "$OUT/game-test" "$HERE/main.swift" "${SOURCES[@]}"
"$OUT/game-test"
