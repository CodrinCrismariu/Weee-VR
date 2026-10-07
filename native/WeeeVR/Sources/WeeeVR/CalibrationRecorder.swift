import CoreVideo
import Foundation
import QuartzCore
import UIKit

/// Records a camera-IMU calibration dataset for Kalibr (scripts/calib/): the exact 640×480 gray
/// images XRSLAM tracks on (ultra-wide, GDC on, same downscale) at 20 Hz, and raw gyro/accelerometer
/// at 200 Hz, all on the host clock. Written to Documents/calibration/<date>/:
///   frames.bin  per frame: Float64 time, UInt32 width, UInt32 height, width*height bytes
///   gyro.csv    time,x,y,z  (rad/s, device frame, raw)
///   accel.csv   time,x,y,z  (g, device frame, raw CoreMotion sign: -1 on z when lying face up)
///   meta.json   camera format, intrinsics reported by AVFoundation, device
final class CalibrationRecorder {
    static let warmup = 3.0             // s before recording starts (point the camera at the target)
    static let duration = 75.0          // s recorded
    static let frameEvery = 3           // 60 fps camera -> 20 Hz images

    private let motion: MotionTracker
    private let queue = DispatchQueue(label: "calibration.writer", qos: .utility)
    private let imuQueue = OperationQueue()

    private let lock = NSLock()
    private var active = false
    private var startTime = 0.0
    private var directory: URL?
    private var frames: FileHandle?
    private var gyroLines: [String] = []
    private var accelLines: [String] = []
    private var frameCount = 0
    private var writtenFrames = 0
    private var droppedFrames = 0
    private var pendingWrites = 0
    private var meta: [String: Any] = [:]
    private var onFinish: ((URL?) -> Void)?

    init(motion: MotionTracker) {
        self.motion = motion
        imuQueue.underlyingQueue = queue
        imuQueue.maxConcurrentOperationCount = 1
    }

    var isRecording: Bool {
        lock.lock()
        defer { lock.unlock() }
        return active
    }

    /// HUD line while recording.
    var status: String? {
        lock.lock()
        defer { lock.unlock() }
        guard active else { return nil }
        let elapsed = CACurrentMediaTime() - startTime
        if elapsed < 0 { return String(format: "CALIBRATION: aim at the grid… %.0f", -elapsed) }
        return String(format: "REC %.0f / %.0f s · %ld frames · drop %ld", elapsed, Self.duration, writtenFrames,
                      droppedFrames)
    }

    func start(onFinish: @escaping (URL?) -> Void) {
        let stamp = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-")
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let root = documents.appendingPathComponent("calibration", isDirectory: true)
        let dir = root.appendingPathComponent(stamp, isDirectory: true)
        // A recording is ~450 MB: keep only the newest one, and don't start without room for it.
        try? FileManager.default.removeItem(at: root)
        let free = (try? documents.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]))?
            .volumeAvailableCapacityForImportantUsage ?? 0
        guard free > 1_500_000_000 else {
            Log.info("calibration: not enough free storage (\(free / 1_000_000) MB)")
            onFinish(nil)
            return
        }
        do {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            FileManager.default.createFile(atPath: dir.appendingPathComponent("frames.bin").path, contents: nil)
            let handle = try FileHandle(forWritingTo: dir.appendingPathComponent("frames.bin"))
            lock.lock()
            active = true
            startTime = CACurrentMediaTime() + Self.warmup
            directory = dir
            frames = handle
            gyroLines = ["time,x,y,z"]
            accelLines = ["time,x,y,z"]
            frameCount = 0
            writtenFrames = 0
            droppedFrames = 0
            meta = ["device": Self.deviceModel(), "started": stamp, "image": [XRSLAMSource.width, XRSLAMSource.height],
                    "frameRate": 60 / Self.frameEvery, "imuRate": 200]
            self.onFinish = onFinish
            lock.unlock()
        } catch {
            Log.info("calibration: cannot create \(dir.path): \(error.localizedDescription)")
            onFinish(nil)
            return
        }

        motion.startRawIMU(interval: 0.005, queue: imuQueue, gyro: { [weak self] t, w in
            self?.addIMU(t, w, gyro: true)
        }, accel: { [weak self] t, a in
            self?.addIMU(t, a, gyro: false)
        })
        DispatchQueue.main.asyncAfter(deadline: .now() + Self.warmup + Self.duration) { [weak self] in self?.finish() }
        Log.info("calibration: recording to \(dir.path)")
    }

    /// Capture queue, every camera frame.
    func add(_ frame: CameraFrame) {
        lock.lock()
        guard active, frame.captureTime >= startTime else { lock.unlock(); return }
        frameCount += 1
        guard frameCount % Self.frameEvery == 0 else { lock.unlock(); return }
        if pendingWrites > 4 {
            droppedFrames += 1
            lock.unlock()
            return
        }
        pendingWrites += 1
        if meta["intrinsics"] == nil {
            meta["sourceSize"] = [frame.size.x, frame.size.y]
            meta["intrinsics"] = [frame.intrinsics.x, frame.intrinsics.y, frame.intrinsics.z, frame.intrinsics.w]
        }
        lock.unlock()

        let buffer = frame.pixelBuffer, time = frame.captureTime
        queue.async { [self] in
            var record = Data(count: 16 + XRSLAMSource.width * XRSLAMSource.height)
            let ok = record.withUnsafeMutableBytes { bytes -> Bool in
                bytes.storeBytes(of: time, toByteOffset: 0, as: Double.self)
                bytes.storeBytes(of: UInt32(XRSLAMSource.width), toByteOffset: 8, as: UInt32.self)
                bytes.storeBytes(of: UInt32(XRSLAMSource.height), toByteOffset: 12, as: UInt32.self)
                return XRSLAMSource.downscaleLuma(buffer, into: bytes.baseAddress! + 16)
            }
            lock.lock()
            pendingWrites -= 1
            let handle = active ? frames : nil
            if ok, handle != nil { writtenFrames += 1 }
            lock.unlock()
            if ok { handle?.write(record) }
        }
    }

    private func addIMU(_ t: Double, _ v: SIMD3<Double>, gyro: Bool) {
        lock.lock()
        defer { lock.unlock() }
        guard active, t >= startTime - 0.5 else { return }    // a little IMU before the first image
        let line = String(format: "%.9f,%.9f,%.9f,%.9f", t, v.x, v.y, v.z)
        if gyro { gyroLines.append(line) } else { accelLines.append(line) }
    }

    func finish() {
        motion.stopRawIMU()
        queue.async { [self] in       // after the queued frame writes
            lock.lock()
            guard active, let dir = directory else { lock.unlock(); return }
            active = false
            let gyro = gyroLines.joined(separator: "\n") + "\n"
            let accel = accelLines.joined(separator: "\n") + "\n"
            meta["frames"] = writtenFrames
            meta["droppedFrames"] = droppedFrames
            meta["gyroSamples"] = gyroLines.count - 1
            meta["accelSamples"] = accelLines.count - 1
            let info = meta
            try? frames?.close()
            frames = nil
            let done = onFinish
            onFinish = nil
            lock.unlock()

            try? gyro.write(to: dir.appendingPathComponent("gyro.csv"), atomically: true, encoding: .utf8)
            try? accel.write(to: dir.appendingPathComponent("accel.csv"), atomically: true, encoding: .utf8)
            if let json = try? JSONSerialization.data(withJSONObject: info, options: [.prettyPrinted, .sortedKeys]) {
                try? json.write(to: dir.appendingPathComponent("meta.json"))
            }
            Log.info(String(format: "calibration: finished %@ frames=%ld dropped=%ld gyro=%ld accel=%ld",
                            dir.lastPathComponent, info["frames"] as? Int ?? 0, info["droppedFrames"] as? Int ?? 0,
                            info["gyroSamples"] as? Int ?? 0, info["accelSamples"] as? Int ?? 0))
            DispatchQueue.main.async { done?(dir) }
        }
    }

    private static func deviceModel() -> String {
        var info = utsname()
        uname(&info)
        return withUnsafeBytes(of: &info.machine) { String(decoding: $0.prefix { $0 != 0 }, as: UTF8.self) }
    }
}
