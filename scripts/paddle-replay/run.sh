#!/usr/bin/env bash
# Replay a smart-paddle recording through the app's PaddleFusion on Linux (WSL):
#   scripts/paddle-replay/run.sh <recording.csv>
# Recordings come from the app (Settings > Smart paddle > Record paddle data) in Documents/paddle-rec/.
set -e
source ~/.local/share/swiftly/env.sh
HERE="$(cd "$(dirname "$0")" && pwd)"
GAME="$HERE/../../native/WeeeVR/Sources/WeeeVR/Game"
OUT="${TMPDIR:-$HOME}/weee-paddle-replay"
mkdir -p "$OUT"
SOURCES=()
for f in "$GAME"/*.swift; do [[ "$f" == *GameAudio* ]] || SOURCES+=("$f"); done
swiftc -O -module-name PaddleReplay -o "$OUT/paddle-replay" "$HERE/main.swift" "${SOURCES[@]}"
"$OUT/paddle-replay" "$@"
