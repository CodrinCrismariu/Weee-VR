#!/usr/bin/env bash
# Camera + IMU calibration of the ultra-wide tracking images, end to end. Run inside WSL:
#   scripts/calib/run.sh <tag-size-mm> [recording-name | local recording dir]
# 1. pulls the newest (or named) recording from the phone, 2. converts it for Kalibr,
# 3. runs Kalibr (intrinsics with pinhole-radtan, then camera-IMU extrinsics + time offset) in
# Docker, 4. writes the result into Sources/WeeeVR/UltraWideCalibration.swift.
# One-time setup: Docker image `kalibr` (github.com/ethz-asl/kalibr, Dockerfile_ros1_20_04) and
# ~/calib/venv with pymobiledevice3 numpy pillow pyyaml.
set -euo pipefail

TAG_MM=${1:?usage: run.sh <tag-size-mm> [recording-name]}
NAME=${2:-}
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$HOME/calib/venv/bin/python"
WORK="$HOME/calib/runs/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$WORK"

if [ -n "$NAME" ] && [ -d "$NAME" ]; then
    REC="$NAME"
else
    echo "== pull"
    "$PY" "$REPO/scripts/calib/pull.py" "$WORK/raw" $NAME
    REC="$(ls -d "$WORK"/raw/*/ | head -1)"
fi

echo "== convert"
"$PY" "$REPO/scripts/calib/to_kalibr.py" "$REC" "$WORK/dataset"

# AprilGrid shown by tools/aprilgrid.html: 10 x 6 tags, gap = 0.3 x tag size (sized for a large monitor).
cat > "$WORK/aprilgrid.yaml" <<EOF
target_type: 'aprilgrid'
tagCols: 10
tagRows: 6
tagSize: $(python3 -c "print($TAG_MM / 1000)")
tagSpacing: 0.3
EOF
# Phone IMU noise (typical MEMS values, inflated as Kalibr recommends for real-world data).
# CoreMotion delivers raw IMU at ~100 Hz on the iPhone 15 even when asked for 200.
IMU_RATE=$(awk -F, 'NR==2{a=$1} END{print (NR-2)/(($1-a)/1e9)}' "$WORK/dataset/imu0.csv")
cat > "$WORK/imu.yaml" <<EOF
rostopic: /imu0
update_rate: $IMU_RATE
accelerometer_noise_density: 2.0e-3
accelerometer_random_walk: 4.0e-4
gyroscope_noise_density: 2.0e-4
gyroscope_random_walk: 4.0e-5
EOF

echo "== kalibr (several minutes)"
DOCKER=docker
$DOCKER info >/dev/null 2>&1 || DOCKER="sudo docker"   # until the docker group applies
$DOCKER run --rm -v "$WORK:/data" -w /data -e MPLBACKEND=Agg -e KALIBR_MANUAL_FOCAL_LENGTH_INIT=1 \
    --entrypoint bash kalibr -c '
    set -e
    source /catkin_ws/devel/setup.bash
    rosrun kalibr kalibr_bagcreater --folder /data/dataset --output-bag /data/calib.bag
    rosrun kalibr kalibr_calibrate_cameras --bag /data/calib.bag --topics /cam0/image_raw \
        --models pinhole-radtan --target /data/aprilgrid.yaml --bag-freq 4.0 --dont-show-report
    rosrun kalibr kalibr_calibrate_imu_camera --bag /data/calib.bag --cam /data/calib-camchain.yaml \
        --imu /data/imu.yaml --target /data/aprilgrid.yaml --dont-show-report
' 2>&1 | tee "$WORK/kalibr.log" | grep -E -i "reprojection|error|timeshift|Calibration|^T_|Initializ|Processed" || true

echo "== apply"
"$PY" "$REPO/scripts/calib/apply.py" "$WORK/calib-camchain-imucam.yaml" "$REC/meta.json" \
    "$REPO/native/WeeeVR/Sources/WeeeVR/UltraWideCalibration.swift"
echo "== reports in $WORK (calib-report-cam.pdf, calib-report-imucam.pdf, *results*.txt)"
