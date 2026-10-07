import CoreMotion
import Foundation
import simd

/// Buffers CoreMotion attitude samples so we can ask "what was the head orientation at time t?"
/// for any t: interpolated for the past (camera capture time), gyro-extrapolated for the future
/// (display time). Timestamps share the host clock with camera/ARKit frames and CACurrentMediaTime.
final class MotionTracker {
    private struct Sample {
        var time: Double
        var attitude: simd_quatd        // world(z-up reference) <- device
        var rate: SIMD3<Double>         // rad/s, device frame, bias-corrected
    }

    private let manager = CMMotionManager()
    private let queue = OperationQueue()
    private let lock = NSLock()
    private var samples: [Sample] = []

    func start() {
        guard manager.isDeviceMotionAvailable, !manager.isDeviceMotionActive else { return }
        queue.maxConcurrentOperationCount = 1
        queue.qualityOfService = .userInteractive
        manager.deviceMotionUpdateInterval = 1.0 / 200.0
        manager.startDeviceMotionUpdates(using: .xArbitraryZVertical, to: queue) { [weak self] motion, _ in
            guard let self, let motion else { return }
            let q = motion.attitude.quaternion
            let r = motion.rotationRate
            let sample = Sample(
                time: motion.timestamp,
                attitude: simd_quatd(ix: q.x, iy: q.y, iz: q.z, r: q.w),
                rate: SIMD3(r.x, r.y, r.z)
            )
            self.lock.lock()
            self.samples.append(sample)
            if self.samples.count > 400 { self.samples.removeFirst(self.samples.count - 300) }
            self.lock.unlock()
        }
    }

    /// Raw (uncalibrated) gyroscope in rad/s and accelerometer in g, device frame, for visual-inertial
    /// tracking. Shares this CMMotionManager: Apple advises one per app.
    func startRawIMU(interval: Double, queue: OperationQueue,
                     gyro: @escaping (Double, SIMD3<Double>) -> Void,
                     accel: @escaping (Double, SIMD3<Double>) -> Void) {
        guard manager.isGyroAvailable, manager.isAccelerometerAvailable else { return }
        manager.gyroUpdateInterval = interval
        manager.accelerometerUpdateInterval = interval
        manager.startGyroUpdates(to: queue) { data, _ in
            guard let data else { return }
            gyro(data.timestamp, SIMD3(data.rotationRate.x, data.rotationRate.y, data.rotationRate.z))
        }
        manager.startAccelerometerUpdates(to: queue) { data, _ in
            guard let data else { return }
            accel(data.timestamp, SIMD3(data.acceleration.x, data.acceleration.y, data.acceleration.z))
        }
    }

    func stopRawIMU() {
        manager.stopGyroUpdates()
        manager.stopAccelerometerUpdates()
    }

    /// Device attitude (world <- device) at host time `t`.
    func attitude(at t: Double) -> simd_quatd? {
        lock.lock()
        defer { lock.unlock() }
        guard let newest = samples.last else { return nil }

        if t >= newest.time {
            // Predict forward with the body-frame rotation rate (capped to avoid runaway).
            let dt = min(t - newest.time, 0.1)
            let angle = simd_length(newest.rate) * dt
            guard angle > 1e-9 else { return newest.attitude }
            let step = simd_quatd(angle: angle, axis: simd_normalize(newest.rate))
            return simd_normalize(newest.attitude * step)
        }

        var i = samples.count - 1
        while i > 0, samples[i - 1].time > t { i -= 1 }
        guard i > 0 else { return samples[0].attitude }
        let a = samples[i - 1], b = samples[i]
        let f = (t - a.time) / max(b.time - a.time, 1e-6)
        return simd_slerp(a.attitude, b.attitude, f)
    }

    var latestTime: Double? {
        lock.lock()
        defer { lock.unlock() }
        return samples.last?.time
    }
}
