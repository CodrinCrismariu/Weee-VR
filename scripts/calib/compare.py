"""Compare app calibration recordings (e.g. before and after an iOS update) on the raw sensor data.

usage: python compare.py <recording-dir> [<recording-dir> ...]      (~/calib/venv/bin/python in WSL)

Per recording:
  imu     gyro/accel rate, sample interval median / p99 / max, gaps > 20 ms, non-increasing stamps
  noise   gyro and accel white noise from sample-to-sample differences in the calmest 10% of the
          recording, accel |a| there (should be ~1 g)
  camera  frame interval median / max, brightness, sharpness (Laplacian variance, lower = blurrier)
  timing  camera-IMU time offset estimated independently of Kalibr: image motion between frames
          (phase correlation) against gyro rotation rate around the image plane, best lag over
          +-60 ms, as Kalibr's convention: the image matches the gyro at t_cam + shift (Kalibr's
          iOS 26 calibration: +2.8 ms).
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np

G = 9.80665


def load_imu(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", skiprows=1)


def frames(path: Path, step: int = 4):
    """(time, image downscaled by `step`) for each frame."""
    with open(path, "rb") as f:
        while header := f.read(16):
            t, w, h = struct.unpack("<dII", header)
            pixels = np.frombuffer(f.read(w * h), dtype=np.uint8).reshape(h, w)
            yield t, pixels[::step, ::step].astype(np.float32)


def intervals(t: np.ndarray) -> str:
    d = np.diff(t) * 1000
    gap = 1.5 * np.median(d)
    return (f"dt median {np.median(d):.2f} p99 {np.percentile(d, 99):.2f} max {d.max():.1f} ms, "
            f"gaps>{gap:.0f}ms {(d > gap).sum()}, non-increasing {(d <= 0).sum()}")


def calm_noise(samples: np.ndarray, rate: float) -> tuple[float, float]:
    """White-noise density from first differences in the calmest windows (per axis, mean), and the
    mean magnitude there."""
    diff = np.diff(samples, axis=0) / np.sqrt(2)
    window = max(int(rate), 10)
    energy = np.convolve((diff ** 2).sum(axis=1), np.ones(window) / window, mode="valid")
    calm = energy <= np.percentile(energy, 10)
    idx = np.nonzero(calm)[0]
    sigma = np.sqrt(np.mean(diff[idx] ** 2))
    magnitude = np.linalg.norm(samples[idx], axis=1).mean()
    return sigma / np.sqrt(rate), magnitude      # noise density [unit/sqrt(Hz)]


def phase_shift(a: np.ndarray, b: np.ndarray) -> float:
    """Image translation magnitude between two frames (pixels of the downscaled image)."""
    window = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    fa = np.fft.fft2((a - a.mean()) * window)
    fb = np.fft.fft2((b - b.mean()) * window)
    cross = fa * np.conj(fb)
    cross /= np.abs(cross) + 1e-9
    corr = np.fft.ifft2(cross).real
    y, x = np.unravel_index(np.argmax(corr), corr.shape)
    if y > a.shape[0] // 2: y -= a.shape[0]
    if x > a.shape[1] // 2: x -= a.shape[1]
    return float(np.hypot(x, y))


def main(dirs: list[Path]) -> None:
    for d in dirs:
        meta = json.loads((d / "meta.json").read_text())
        gyro, accel = load_imu(d / "gyro.csv"), load_imu(d / "accel.csv")
        span = gyro[-1, 0] - gyro[0, 0]
        print(f"== {d.name}  ({meta.get('device')}, {meta.get('frames')} frames, dropped {meta.get('droppedFrames')}, "
              f"intrinsics {meta.get('intrinsics')})")
        g_rate, a_rate = len(gyro) / span, len(accel) / (accel[-1, 0] - accel[0, 0])
        print(f"  gyro  {g_rate:6.1f} Hz  {intervals(gyro[:, 0])}")
        print(f"  accel {a_rate:6.1f} Hz  {intervals(accel[:, 0])}")
        g_noise, g_mag = calm_noise(gyro[:, 1:], g_rate)
        a_noise, a_mag = calm_noise(accel[:, 1:] * G, a_rate)
        print(f"  noise gyro {g_noise:.2e} rad/s/sqrtHz (calm |w| {g_mag:.3f} rad/s), "
              f"accel {a_noise:.2e} m/s2/sqrtHz (calm |a| {a_mag:.3f} m/s2 = {a_mag / G:.4f} g)")

        times, shifts, bright, sharp, stamps = [], [], [], [], []
        previous = None
        for t, image in frames(d / "frames.bin"):
            stamps.append(t)
            bright.append(image.mean())
            lap = image[1:-1, 1:-1] * 4 - image[:-2, 1:-1] - image[2:, 1:-1] - image[1:-1, :-2] - image[1:-1, 2:]
            sharp.append(lap.var())
            if previous is not None:
                times.append((previous[0] + t) / 2)
                shifts.append(phase_shift(previous[1], image) / (t - previous[0]))
            previous = (t, image)
        frame_t, shifts = np.array(times), np.array(shifts)      # frame-pair midpoints
        print(f"  camera {len(stamps) / (stamps[-1] - stamps[0]):5.1f} Hz  {intervals(np.array(stamps))}")
        print(f"  image brightness {np.median(bright):.1f}, sharpness median {np.median(sharp):.1f} "
              f"p10 {np.percentile(sharp, 10):.1f}")

        # Gyro rotation around the image plane (device x, y; the optical axis is device z).
        rate = np.hypot(gyro[:, 1], gyro[:, 2])
        lags = np.arange(-0.060, 0.0605, 0.0005)
        valid = (frame_t > gyro[0, 0] + 0.1) & (frame_t < gyro[-1, 0] - 0.1) & (shifts < np.percentile(shifts, 98))
        scores = []
        for lag in lags:
            g = np.interp(frame_t[valid] - lag, gyro[:, 0], rate)
            scores.append(np.corrcoef(g, shifts[valid])[0, 1])
        scores = np.array(scores)
        best = lags[np.argmax(scores)]
        # Sub-sample peak by a parabola through the best three.
        i = int(np.argmax(scores))
        if 0 < i < len(scores) - 1:
            a, b, c = scores[i - 1], scores[i], scores[i + 1]
            best += 0.0005 * 0.5 * (a - c) / (a - 2 * b + c)
        print(f"  timing camera-IMU shift {-best * 1000:+.1f} ms (correlation {scores.max():.3f})")


if __name__ == "__main__":
    main([Path(p) for p in sys.argv[1:]])
