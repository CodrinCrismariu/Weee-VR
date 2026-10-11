import CoreVideo
import Foundation
import QuartzCore
import XRSLAMCore
import simd

/// Finds the smart paddle's AprilTags (tagStandard41h12) in the passthrough camera frames and
/// estimates their poses, on its own queue (a frame is skipped while the previous one is still being
/// processed): ID 0 (69.4 mm black square) on the backhand face, 68 and 69 (27.8 mm) on the forehand.
///
/// While the paddle is tracked only a window around it is searched (where the IMU says it is, else
/// where its tags were), at full resolution when small, which is cheap; otherwise the whole frame
/// with quads found at half resolution. Fast swings blur the tags away (the IMU carries the paddle
/// then); they are found again as soon as the paddle slows.
final class TagTracker {
    static let tags: [(id: Int32, size: Double)] = [(0, 0.0694), (68, 0.0278), (69, 0.0278)]
    static let bigTagSize = 0.0694

    struct Solution {
        var rotation: simd_double3x3            // view <- tag (view: x right, y up, -z forward)
        var translation: SIMD3<Double>          // tag centre, view frame, m
        var error: Double
    }

    struct Tag {
        var id: Int
        var size: Double                        // black square, m
        var solutions: [Solution]               // best fit first
        var margin: Float
    }

    struct Detection {
        var sequence: UInt64
        var captureTime: Double
        var pose: simd_float4x4?                // world <- view at capture (6DoF modes)
        var tags: [Tag]
    }

    private let queue = DispatchQueue(label: "paddle.tag", qos: .userInitiated)
    private let lock = NSLock()
    private var busy = false
    private var _enabled = false
    private var _flipped = false
    private var latest: Detection?
    private var _hint: (centre: SIMD3<Float>, radius: Float, time: Double)?
    private var sequence: UInt64 = 0
    private var processMS = 0.0
    private var rateCount = 0, foundCount = 0
    private var rateStart = CACurrentMediaTime()
    private var rate = 0.0, foundRate = 0.0

    // `queue` only
    private var detector: OpaquePointer?
    private var window: (centre: SIMD2<Double>, size: Double, time: Double)?
    private var reportedIDs: Set<Int> = []

    deinit {
        if let detector { weee_tag_destroy(detector) }
    }

    var enabled: Bool {
        get { lock.lock(); defer { lock.unlock() }; return _enabled }
        set {
            lock.lock()
            _enabled = newValue
            if !newValue { latest = nil }
            lock.unlock()
        }
    }

    /// The camera image is upside down (phone the other way round in the viewer).
    var flipped: Bool {
        get { lock.lock(); defer { lock.unlock() }; return _flipped }
        set { lock.lock(); _flipped = newValue; lock.unlock() }
    }

    /// Capture queue: hand the frame over unless the previous one is still being processed.
    func submit(_ frame: CameraFrame) {
        lock.lock()
        guard _enabled, !busy else { lock.unlock(); return }
        busy = true
        let flipped = _flipped
        let hint = _hint
        lock.unlock()
        queue.async { [self] in
            let start = CACurrentMediaTime()
            let found = detect(frame, flipped: flipped, hint: hint)
            let elapsed = (CACurrentMediaTime() - start) * 1000
            lock.lock()
            busy = false
            if _enabled, !found.isEmpty {
                sequence += 1
                latest = Detection(sequence: sequence, captureTime: frame.captureTime, pose: frame.pose, tags: found)
                foundCount += 1
            }
            processMS += (elapsed - processMS) * 0.1
            rateCount += 1
            let now = CACurrentMediaTime()
            if now - rateStart >= 1 {
                rate = Double(rateCount) / (now - rateStart)
                foundRate = Double(foundCount) / (now - rateStart)
                rateCount = 0
                foundCount = 0
                rateStart = now
            }
            lock.unlock()
        }
    }

    /// Render thread: where the IMU says the blade centre is (world, in the frames' own pose
    /// coordinates, i.e. without the render thread's position offset) at host time `time`, and a radius
    /// holding all its tags. Searched first.
    func setHint(centre: SIMD3<Float>, radius: Float, time: Double) {
        lock.lock()
        _hint = (centre, radius, time)
        lock.unlock()
    }

    /// Render thread: the newest sighting (check `sequence` to see whether it is new).
    func latestDetection() -> Detection? {
        lock.lock()
        defer { lock.unlock() }
        return latest
    }

    /// (ms per processed frame, frames processed per second, frames with a paddle tag per second)
    var stats: (ms: Double, hz: Double, foundHz: Double) {
        lock.lock()
        defer { lock.unlock() }
        return (processMS, rate, foundRate)
    }

    // MARK: Detection (queue)

    /// Intrinsics and radtan distortion in this frame's pixels: the Kalibr calibration when it is the
    /// ultra-wide's 1440x1080 frame (the calibration is of its 640x480 downscale), else the frame's own.
    private static func camera(for frame: CameraFrame) -> ([Double], [Double]) {
        let c = UltraWideCalibration.self
        let k = frame.intrinsics
        if c.available, frame.size == SIMD2(1440, 1080) {
            let s = 1440.0 / Double(c.width)
            let fx = c.fx * s
            // The same lens? Its focal length agrees with the frame's (a 1x frame would differ a lot).
            if abs(fx / Double(k.x) - 1) < 0.1 {
                return ([fx, c.fy * s, (c.cx + 0.5) * s - 0.5, (c.cy + 0.5) * s - 0.5],
                        [c.distortion.0, c.distortion.1, c.distortion.2, c.distortion.3])
            }
        }
        return ([Double(k.x), Double(k.y), Double(k.z), Double(k.w)], [0, 0, 0, 0])
    }

    /// The hinted blade centre in this frame's pixels, and the hint's diameter there (px).
    private static func project(_ hint: (centre: SIMD3<Float>, radius: Float, time: Double)?, into frame: CameraFrame,
                                flipped: Bool) -> (SIMD2<Double>, Double)? {
        guard let hint, abs(hint.time - frame.captureTime) < 0.15, let pose = frame.pose else { return nil }
        let r = pose.rotation.float, t = pose.translation
        let v = r.transpose * (hint.centre - t)                 // view frame: -z forward
        guard v.z < -0.08 else { return nil }
        let k = frame.intrinsics
        var px = SIMD2<Double>(Double(k.z + k.x * v.x / -v.z), Double(k.w - k.y * v.y / -v.z))
        if flipped { px = SIMD2(Double(frame.size.x), Double(frame.size.y)) - px }
        return (px, Double(k.x * 2 * hint.radius / -v.z))
    }

    private func detect(_ frame: CameraFrame, flipped: Bool,
                        hint: (centre: SIMD3<Float>, radius: Float, time: Double)?) -> [Tag] {
        if detector == nil { detector = weee_tag_create(2) }
        guard let detector else { return [] }
        let buffer = frame.pixelBuffer
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddressOfPlane(buffer, 0) else { return [] }
        let width = CVPixelBufferGetWidthOfPlane(buffer, 0)
        let height = CVPixelBufferGetHeightOfPlane(buffer, 0)
        let stride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 0)
        let (camera, distortion) = Self.camera(for: frame)
        let ids = Self.tags.map { Int32($0.id) }, sizes = Self.tags.map(\.size)

        // Where to look: around the paddle (where the IMU puts it, else where its tags were, with room
        // for it to move between frames); if nothing is there, the whole frame at once.
        var outs = [WeeeTagDetection](repeating: WeeeTagDetection(), count: 3)
        func search(_ x0: Int, _ y0: Int, _ w: Int, _ h: Int, decimate: Float) -> Int {
            camera.withUnsafeBufferPointer { k in
                distortion.withUnsafeBufferPointer { d in
                    ids.withUnsafeBufferPointer { i in
                        sizes.withUnsafeBufferPointer { z in
                            Int(weee_tag_detect(detector, base.assumingMemoryBound(to: UInt8.self), Int32(stride),
                                                Int32(x0), Int32(y0), Int32(w), Int32(h), decimate, i.baseAddress,
                                                z.baseAddress, Int32(ids.count), k.baseAddress, d.baseAddress, &outs,
                                                Int32(outs.count)))
                        }
                    }
                }
            }
        }
        func searchAround(_ centre: SIMD2<Double>, size: Double) -> Int {
            let half = max(size * 0.75 + 90, 160)
            let x0 = max(Int(centre.x - half), 0), y0 = max(Int(centre.y - half), 0)
            let w = min(Int(centre.x + half), width) - x0, h = min(Int(centre.y + half), height) - y0
            guard w >= 32, h >= 32 else { return 0 }
            return search(x0, y0, w, h, decimate: max(w, h) > 640 ? 2 : 1)
        }
        var found = 0
        let recent = window.flatMap { frame.captureTime - $0.time < 0.3 ? $0 : nil }
        let predicted = Self.project(hint, into: frame, flipped: flipped)
        if let predicted { found = searchAround(predicted.0, size: predicted.1) }
        if found == 0, let recent, predicted.map({ simd_distance($0.0, recent.centre) > recent.size * 0.3 }) ?? true {
            found = searchAround(recent.centre, size: recent.size)
        }
        if found == 0 { found = search(0, 0, width, height, decimate: 2) }
        guard found > 0 else {
            window = nil
            return []
        }

        // OpenCV camera (x right, y down, z forward) -> view (x right, y up, -z forward); an upside-down
        // image is the camera turned half a turn about its axis.
        let toView = flipped ? simd_double3x3(diagonal: SIMD3(-1, 1, -1)) : simd_double3x3(diagonal: SIMD3(1, -1, -1))
        var tags: [Tag] = []
        var lo = SIMD2<Double>(repeating: .infinity), hi = SIMD2<Double>(repeating: -.infinity)
        for out in outs.prefix(found) {
            let corners = withUnsafeBytes(of: out.corners) { Array($0.bindMemory(to: Double.self)) }
            for c in 0..<4 {
                lo = simd_min(lo, SIMD2(corners[2 * c], corners[2 * c + 1]))
                hi = simd_max(hi, SIMD2(corners[2 * c], corners[2 * c + 1]))
            }
            let rotations = withUnsafeBytes(of: out.rotation) { Array($0.bindMemory(to: Double.self)) }
            let translations = withUnsafeBytes(of: out.translation) { Array($0.bindMemory(to: Double.self)) }
            let errors = withUnsafeBytes(of: out.error) { Array($0.bindMemory(to: Double.self)) }
            var solutions: [Solution] = []
            for s in 0..<Int(out.solutions) {
                let r = Array(rotations[(9 * s)..<(9 * s + 9)])
                let rotation = simd_double3x3(rows: [SIMD3(r[0], r[1], r[2]), SIMD3(r[3], r[4], r[5]), SIMD3(r[6], r[7], r[8])])
                let t = SIMD3(translations[3 * s], translations[3 * s + 1], translations[3 * s + 2])
                solutions.append(Solution(rotation: toView * rotation, translation: toView * t, error: errors[s]))
            }
            solutions.sort { $0.error < $1.error }
            let size = Self.tags.first { $0.id == out.id }?.size ?? Self.bigTagSize
            if !solutions.isEmpty { tags.append(Tag(id: Int(out.id), size: size, solutions: solutions, margin: out.margin)) }
        }
        // The window: around all the tags seen, as big as the blade would be (a small tag is a fifth
        // of it), so the other tags and the next frame's move fit.
        let extent = simd_reduce_max(hi - lo)
        let blade = tags.contains { $0.id == 0 } ? extent * 2.5 : extent * 1.6
        window = ((lo + hi) / 2, max(blade, 120), frame.captureTime)

        let seen = Set(tags.map(\.id))
        if !seen.isSubset(of: reportedIDs) {
            reportedIDs.formUnion(seen)
            Log.info("tag: seen IDs \(reportedIDs.sorted())")
        }
        return tags
    }
}
