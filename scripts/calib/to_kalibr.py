"""Convert an app calibration recording into Kalibr's bagcreater folder layout.

usage: python to_kalibr.py <recording-dir> <out-dir>
  out-dir/cam0/<ns>.png   640x480 gray images
  out-dir/imu0.csv        timestamp [ns], gyro x y z [rad/s], accel x y z [m/s^2]
Gyro and accelerometer arrive as separate CoreMotion streams; accel is interpolated onto the gyro
timestamps. CoreMotion reports -1 g on z lying face up, so accel * -9.80665 gives the specific force
(+9.81 up at rest) that Kalibr expects. Timestamps stay on the device host clock (seconds -> ns).
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image

G = 9.80665


def main(src: Path, out: Path) -> None:
    meta = json.loads((src / "meta.json").read_text())
    (out / "cam0").mkdir(parents=True, exist_ok=True)

    count = 0
    first = last = None
    with open(src / "frames.bin", "rb") as f:
        while header := f.read(16):
            t, w, h = struct.unpack("<dII", header)
            pixels = f.read(w * h)
            if len(pixels) < w * h:
                break
            Image.frombytes("L", (w, h), pixels).save(out / "cam0" / f"{round(t * 1e9)}.png")
            first = t if first is None else first
            last = t
            count += 1

    gyro = np.loadtxt(src / "gyro.csv", delimiter=",", skiprows=1)
    accel = np.loadtxt(src / "accel.csv", delimiter=",", skiprows=1)
    gyro = gyro[(gyro[:, 0] >= accel[0, 0]) & (gyro[:, 0] <= accel[-1, 0])]
    a = np.stack([np.interp(gyro[:, 0], accel[:, 0], accel[:, i]) for i in (1, 2, 3)], axis=1) * -G
    with open(out / "imu0.csv", "w") as f:
        f.write("timestamp,omega_x,omega_y,omega_z,alpha_x,alpha_y,alpha_z\n")
        for row, acc in zip(gyro, a):
            f.write(f"{round(row[0] * 1e9)},{row[1]:.9f},{row[2]:.9f},{row[3]:.9f},"
                    f"{acc[0]:.9f},{acc[1]:.9f},{acc[2]:.9f}\n")

    rate = (len(gyro) - 1) / (gyro[-1, 0] - gyro[0, 0])
    print(f"{count} images over {last - first:.1f} s, {len(gyro)} IMU rows at {rate:.0f} Hz, "
          f"|accel| median {np.median(np.linalg.norm(a, axis=1)):.2f} m/s^2")
    print("AVFoundation intrinsics at", meta.get("sourceSize"), ":", meta.get("intrinsics"))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
