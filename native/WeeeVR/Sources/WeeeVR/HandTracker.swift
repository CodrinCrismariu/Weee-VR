import CoreML
import CoreVideo
import Foundation
import QuartzCore
import XRSLAMCore
import simd

/// Google MediaPipe Hands on the camera frames: a palm detector finds hands, a landmark model gives
/// 21 joints (x, y in the image, z relative depth) plus presence and handedness for each hand crop.
/// Both are Core ML FP16 models on the Neural Engine (scripts/hands/fetch-models.py), so they never
/// compete with rendering for the GPU.
///
/// Tracking, as MediaPipe does it: once a hand is found, each next frame crops around where its
/// landmarks were, so only the landmark model runs per hand; the palm detector runs only while fewer
/// than two hands are tracked (every frame with none, every 8th with one). Each track keeps its id.
///
/// Left/right is decided once, when a hand is first found, and then locked for the track: from the
/// head-mounted view the model's own left/right output is unreliable (it mostly sees backs of hands),
/// so it is combined with where the wrist is in the image and which way the hand leans (a left hand's
/// fingers point up-right, a right hand's up-left). Two tracked hands never share a side.
///
/// Fast motion (punches): each crop is moved by the hand's image velocity and grown with the
/// distance moved. A hand the landmark model loses (motion blur) coasts for 0.25 s: the palm detector
/// runs at once, and a palm near where the hand was heading takes over its track (same id and side,
/// no re-confirmation), so the rendered hand doesn't drop out or restart.
///
/// False positives: a palm must score 0.75; a new hand must score 0.8 presence on three consecutive
/// frames before it is reported (0.6 keeps a confirmed hand), be within arm's reach (~0.9 m, from its
/// palm's least-foreshortened segment) and point the way the user's own hands can (fingers not down).
///
/// Runs on its own queue; a frame is skipped while the previous one is still being processed.
final class HandTracker {
    static let jointCount = 21
    static let inputSize = 256                 // palm detector input

    struct Hand {
        var id: Int
        var chirality: Int              // 0 left, 1 right (running vote)
        var points: [SIMD2<Float>]      // camera-buffer pixels (origin top-left)
        var depths: [Float]             // relative depth in buffer pixels (smaller = closer)
        var valid: [Bool]
    }

    struct Detection {
        var sequence: UInt64
        var captureTime: Double
        var pose: simd_float4x4?        // world <- camera at capture (6DoF modes)
        var size: SIMD2<Float>
        var intrinsics: SIMD4<Float>
        var hands: [Hand]
    }

    /// A hand crop: square of side `size` (image px) around `centre`, turned by `rotation`
    /// (0 = fingers up in the image).
    private struct Region {
        var centre: SIMD2<Float>
        var size: Float
        var rotation: Float
    }

    private struct Track {
        var id: Int
        var region: Region
        var chirality: Int              // 0 left, 1 right; locked when the track starts
        var wristX: Float
        var lastSeen: Double            // capture time the hand was last found
        var velocity = SIMD2<Float>.zero    // region centre, image px/s
        var age = 1                     // frames found (kept through a coast)

        var confirmed: Bool { age >= HandTracker.confirmFrames }

        /// Where to look at `time`: moved along the hand's velocity, grown by the distance moved
        /// (up to 1.5x) so a mis-predicted fast hand still lands in the crop.
        func predicted(at time: Double) -> Region {
            let shift = velocity * Float(min(max(time - lastSeen, 0), 0.1))
            var r = region
            r.centre += shift
            r.size *= 1 + min(simd_length(shift) / region.size, 0.5)
            return r
        }

        mutating func found(_ next: Region, wristX: Float, at time: Double) {
            let dt = Float(time - lastSeen)
            if dt > 0, dt < 0.3 {
                let v = (next.centre - region.centre) / dt
                velocity = age > 1 ? velocity + (v - velocity) * 0.7 : v
            } else {
                velocity = .zero
            }
            region = next
            self.wristX = wristX
            lastSeen = time
            age += 1
        }
    }

    private static let palmThreshold: Float = 0.75
    private static let newHandPresence: Float = 0.8
    private static let trackingPresence: Float = 0.6
    private static let maxReach: Float = 0.9            // m: farther hands aren't the user's
    private static let maxTilt: Float = 2.2             // rad from fingers-up: the user's hands don't point down
    private static let confirmFrames = 3
    private static let coastTime = 0.25                 // s a lost hand can be picked up again as itself
    private static let maxHands = 2
    private static let maxPalms = 4                     // palm regions considered per detector run

    /// Since launch, for the HUD/log: how a second hand gets found, or why it doesn't.
    struct NewHandCounts {
        var palmRuns = 0            // palm detector runs while one hand was tracked
        var palmMisses = 0          // ... that found no palm away from the tracked hand
        var started = 0             // new-hand tracks started (passed the checks once)
        var confirmed = 0           // ... that passed them on `confirmFrames` frames and were shown
        var rejectPresence = 0      // new-hand checks failed: landmark presence below `newHandPresence`
        var rejectReach = 0         // ... farther than `maxReach`
        var rejectTilt = 0          // ... fingers pointing down

        static func += (a: inout NewHandCounts, b: NewHandCounts) {
            a.palmRuns += b.palmRuns; a.palmMisses += b.palmMisses; a.started += b.started
            a.confirmed += b.confirmed; a.rejectPresence += b.rejectPresence; a.rejectReach += b.rejectReach
            a.rejectTilt += b.rejectTilt
        }
    }

    private let queue = DispatchQueue(label: "hands.mediapipe", qos: .userInitiated)
    private let lock = NSLock()
    private var busy = false
    private var _enabled = true
    private var _flipped = false
    private var latest: Detection?
    private var regained = 0            // lost hands picked up again as themselves
    private var dropped = 0             // confirmed hands lost for good
    private var newHands = NewHandCounts()
    private var sequence: UInt64 = 0
    private var processMS = 0.0
    private var rateCount = 0
    private var rateStart = CACurrentMediaTime()
    private var rate = 0.0

    // `queue` only
    private var detector: MLModel?
    private var landmarker: MLModel?
    private var input: MLMultiArray?            // palm detector input
    private var landmarkInput: MLMultiArray?    // landmark model input, its own size (224 or 256)
    private var tracks: [Track] = []
    private var nextID = 0
    private var framesSinceDetection = 0
    private var frameCounts = NewHandCounts()   // this frame's, added to `newHands` at its end
    private static let anchors = makeAnchors()

    init() {
        queue.async { [self] in loadModels() }
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

    /// The camera image is upside down (phone the other way round in the viewer): "up" for the
    /// fingers-not-down check is image down.
    var flipped: Bool {
        get { lock.lock(); defer { lock.unlock() }; return _flipped }
        set { lock.lock(); _flipped = newValue; lock.unlock() }
    }

    /// Capture queue: hand the frame over unless the previous one is still being processed.
    func submit(_ frame: CameraFrame) {
        lock.lock()
        guard _enabled, !busy else { lock.unlock(); return }
        busy = true
        lock.unlock()
        queue.async { [self] in
            let start = CACurrentMediaTime()
            let hands = process(frame)
            let elapsed = (CACurrentMediaTime() - start) * 1000
            lock.lock()
            busy = false
            sequence += 1
            if _enabled, let hands {
                latest = Detection(sequence: sequence, captureTime: frame.captureTime, pose: frame.pose,
                                   size: frame.size, intrinsics: frame.intrinsics, hands: hands)
            }
            processMS += (elapsed - processMS) * 0.1
            rateCount += 1
            let now = CACurrentMediaTime()
            if now - rateStart >= 1 {
                rate = Double(rateCount) / (now - rateStart)
                rateCount = 0
                rateStart = now
            }
            lock.unlock()
        }
    }

    /// Render thread: the newest detection (check `sequence` to see whether it is new).
    func latestDetection() -> Detection? {
        lock.lock()
        defer { lock.unlock() }
        return latest
    }

    /// (milliseconds per processed frame, processed frames per second, lost hands regained, dropped,
    /// how new hands fared)
    var stats: (ms: Double, hz: Double, regained: Int, dropped: Int, newHands: NewHandCounts) {
        lock.lock()
        defer { lock.unlock() }
        return (processMS, rate, regained, dropped, newHands)
    }

    // MARK: Pipeline (queue)

    private func process(_ frame: CameraFrame) -> [Hand]? {
        guard let detector, let landmarker, input != nil else { return nil }
        let buffer = frame.pixelBuffer
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let image = Image(buffer) else { return nil }

        var hands: [Hand] = []
        var kept: [Track] = []
        var coasting: [Track] = []          // confirmed hands lost this moment, may be picked up again
        var regainedNow = 0, droppedNow = 0
        let now = frame.captureTime
        let focal = frame.intrinsics.x
        let flipped = self.flipped
        func report(_ track: Track, _ hand: (points: [SIMD2<Float>], depths: [Float], valid: [Bool])) {
            hands.append(Hand(id: track.id, chirality: track.chirality, points: hand.points,
                              depths: hand.depths, valid: hand.valid))
        }

        // 1. Tracked hands: landmarks where each hand was heading. Unconfirmed ones are held to the
        //    new-hand checks until confirmed.
        for var track in tracks {
            let found = landmarks(in: track.predicted(at: now), image: image, model: landmarker,
                                  newHand: !track.confirmed, focal: focal, flipped: flipped)
            if let found {
                // Two tracks on one hand: the older one (earlier in `tracks`) keeps it, this one ends.
                // Not coasting: it isn't lost, and a coasting track would hold the second hand's place
                // and keep the palm detector running for nothing.
                if kept.contains(where: { simd_distance($0.region.centre, found.next.centre) < 0.4 * $0.region.size }) {
                    continue
                }
                track.found(found.next, wristX: found.hand.points[0].x, at: now)
                kept.append(track)
                if track.age == Self.confirmFrames { frameCounts.confirmed += 1 }
                if track.confirmed { report(track, found.hand) }
            } else if track.confirmed {
                if now - track.lastSeen < Self.coastTime { coasting.append(track) } else { droppedNow += 1 }
            }
        }

        // 2. Palm detector: at once when a hand was just lost, else while fewer than two are tracked
        //    (every frame with none, every 8th with one).
        framesSinceDetection += 1
        if kept.count < Self.maxHands, kept.isEmpty || !coasting.isEmpty || framesSinceDetection >= 8 {
            framesSinceDetection = 0
            func tracked(_ palm: Region) -> Bool {
                kept.contains(where: { simd_distance($0.region.centre, palm.centre) < 0.5 * $0.region.size })
            }
            let detected = palms(in: image, model: detector)
            if kept.count == 1 {
                frameCounts.palmRuns += 1
                if detected.allSatisfy(tracked) { frameCounts.palmMisses += 1 }
            }
            for palm in detected where kept.count < Self.maxHands {
                if tracked(palm) { continue }       // also hands added earlier in this loop

                // A lost hand showing up near where it was heading: it is that hand.
                let nearest = coasting.indices.min {
                    simd_distance(coasting[$0].predicted(at: now).centre, palm.centre)
                        < simd_distance(coasting[$1].predicted(at: now).centre, palm.centre)
                }
                if let i = nearest,
                   simd_distance(coasting[i].predicted(at: now).centre, palm.centre) < 1.5 * coasting[i].region.size {
                    guard let found = landmarks(in: palm, image: image, model: landmarker, newHand: false,
                                                focal: focal, flipped: flipped) else { continue }
                    var track = coasting.remove(at: i)
                    track.found(found.next, wristX: found.hand.points[0].x, at: now)
                    kept.append(track)
                    report(track, found.hand)
                    regainedNow += 1
                    continue
                }

                // A new hand, while there is room for one (coasting hands keep their place).
                guard kept.count + coasting.count < Self.maxHands,
                      let found = landmarks(in: palm, image: image, model: landmarker, newHand: true,
                                            focal: focal, flipped: flipped) else { continue }
                // Evidence for "right hand", each in -1...1: the model, wrist right of centre, and
                // the hand leaning left (rotation < 0 means fingers point up-left).
                let model = (found.rightness - 0.5) * 2
                let side = min(max((found.hand.points[0].x - Float(image.width) / 2) / (Float(image.width) / 4), -1), 1)
                let lean = min(max(-sin(found.next.rotation) * 2, -1), 1)
                let right = 0.3 * model + 0.4 * side + 0.3 * lean > 0
                var chirality = right ? 1 : 0
                if (kept + coasting).contains(where: { $0.chirality == chirality }) {
                    chirality = 1 - chirality          // the established hand keeps its side
                }
                kept.append(Track(id: nextID, region: found.next, chirality: chirality,
                                  wristX: found.hand.points[0].x, lastSeen: now))
                nextID += 1                 // reported once confirmed
                frameCounts.started += 1
            }
        }
        tracks = kept + coasting
        lock.lock()
        regained += regainedNow
        dropped += droppedNow
        newHands += frameCounts
        lock.unlock()
        frameCounts = NewHandCounts()
        return hands
    }

    /// Landmark model on one region: the hand in image pixels, the region for the next frame, and
    /// P(right hand). Nil when the model says there is no hand; for a `newHand`, also when it isn't
    /// clearly one of the user's own (presence, reach, direction).
    private func landmarks(in region: Region, image: Image, model: MLModel, newHand: Bool, focal: Float, flipped: Bool)
        -> (hand: (points: [SIMD2<Float>], depths: [Float], valid: [Bool]), next: Region, rightness: Float)? {
        let threshold = newHand ? Self.newHandPresence : Self.trackingPresence
        guard let input = landmarkInput else { return nil }
        let size = Float(input.shape[3].intValue)
        let c = cos(region.rotation), s = sin(region.rotation)
        let step = region.size / (size - 1)
        // Crop pixel (u, v) -> image: centre + R * ((u, v) * step - size / 2).
        let half = region.size / 2
        let m: [Float] = [c * step, -s * step, region.centre.x - c * half + s * half,
                          s * step, c * step, region.centre.y - s * half - c * half]
        image.crop(m, size: Int(size), into: input)
        guard let output = try? model.prediction(from: Self.features(input)),
              let presence = output.featureValue(for: "scores")?.multiArrayValue.map({ Self.read($0)[0] })
        else { return nil }
        guard presence >= threshold else {
            if newHand { frameCounts.rejectPresence += 1 }
            return nil
        }
        guard let lr = output.featureValue(for: "lr")?.multiArrayValue.map({ Self.read($0)[0] }),
              let raw = output.featureValue(for: "landmarks")?.multiArrayValue.map({ Self.read($0) }),
              raw.count >= 63
        else { return nil }

        var points = [SIMD2<Float>](repeating: .zero, count: Self.jointCount)
        var depths = [Float](repeating: 0, count: Self.jointCount)
        for i in 0..<Self.jointCount {
            let local = SIMD2(raw[3 * i] - 0.5, raw[3 * i + 1] - 0.5) * region.size
            points[i] = region.centre + SIMD2(c * local.x - s * local.y, s * local.x + c * local.y)
            depths[i] = raw[3 * i + 2] * region.size
        }
        let next = Self.region(fromLandmarks: points)
        if newHand {
            // Within arm's reach? Each palm segment's image length gives a distance; foreshortening
            // (a fist pointing at the camera) only makes one look farther, so the nearest counts.
            var distance = Float.infinity
            for (a, b, length) in HandModel.palmSegments {
                let pixels = simd_distance(points[a], points[b])
                if pixels > 1 { distance = min(distance, length * focal / pixels) }
            }
            guard distance < Self.maxReach else {
                frameCounts.rejectReach += 1
                return nil
            }
            // The user's own hands point up or sideways in their view, not down.
            let up = (next.rotation + (flipped ? Float.pi : 0)).remainder(dividingBy: 2 * .pi)
            guard abs(up) < Self.maxTilt else {
                frameCounts.rejectTilt += 1
                return nil
            }
        }
        let hand = (points: points, depths: depths, valid: [Bool](repeating: true, count: Self.jointCount))
        return (hand, next, lr)
    }

    /// Palm detector on the whole frame (letterboxed to 256x256): regions for the landmark model.
    private func palms(in image: Image, model: MLModel) -> [Region] {
        guard let input else { return [] }
        let size = Float(Self.inputSize)
        let scale = min(size / Float(image.width), size / Float(image.height))
        let pad = (SIMD2(size, size) - SIMD2(Float(image.width), Float(image.height)) * scale) / 2
        image.crop([1 / scale, 0, -pad.x / scale, 0, 1 / scale, -pad.y / scale], size: Self.inputSize, into: input)
        guard let output = try? model.prediction(from: Self.features(input)),
              let coordsArray = output.featureValue(for: "box_coords")?.multiArrayValue,
              let scoresArray = output.featureValue(for: "box_scores")?.multiArrayValue
        else { return [] }
        let coords = Self.read(coordsArray), logits = Self.read(scoresArray)
        let count = Self.anchors.count
        guard coords.count >= count * 18, logits.count >= count else { return [] }

        // Decode candidates above threshold: box + 7 keypoints, in letterbox pixels.
        let logitThreshold = log(Self.palmThreshold / (1 - Self.palmThreshold))
        var candidates: [(score: Float, values: [Float])] = []
        for a in 0..<count where logits[a] >= logitThreshold {
            let anchor = Self.anchors[a] * size
            var values = [Float](repeating: 0, count: 18)
            let cx = coords[a * 18] + anchor.x, cy = coords[a * 18 + 1] + anchor.y
            let w = coords[a * 18 + 2], h = coords[a * 18 + 3]
            values[0] = cx - w / 2; values[1] = cy - h / 2; values[2] = cx + w / 2; values[3] = cy + h / 2
            for k in 0..<7 {
                values[4 + 2 * k] = coords[a * 18 + 4 + 2 * k] + anchor.x
                values[5 + 2 * k] = coords[a * 18 + 5 + 2 * k] + anchor.y
            }
            candidates.append((1 / (1 + exp(-logits[a])), values))
        }

        // Weighted non-maximum suppression (MediaPipe's): merge overlapping boxes by score.
        candidates.sort { $0.score > $1.score }
        // Not capped at `maxHands`: the caller skips palms of hands it already tracks, and a tracked
        // hand (or a stronger false palm) must not use up the slot of the second hand's palm.
        var regions: [Region] = []
        while let best = candidates.first, regions.count < Self.maxPalms {
            var cluster: [(score: Float, values: [Float])] = []
            var rest: [(score: Float, values: [Float])] = []
            for candidate in candidates {
                if Self.iou(best.values, candidate.values) > 0.3 { cluster.append(candidate) } else { rest.append(candidate) }
            }
            candidates = rest
            let total = cluster.reduce(0) { $0 + $1.score }
            var merged = [Float](repeating: 0, count: 18)
            for member in cluster {
                for i in 0..<18 { merged[i] += member.values[i] * member.score / total }
            }
            // Letterbox -> image pixels.
            for i in stride(from: 0, to: 18, by: 2) {
                merged[i] = (merged[i] - pad.x) / scale
                merged[i + 1] = (merged[i + 1] - pad.y) / scale
            }
            let boxSize = max(merged[2] - merged[0], merged[3] - merged[1])
            guard boxSize > 1 else { continue }
            // Keypoint 0 = wrist, 2 = middle finger base: they give the hand's rotation. The hand
            // region is 2.6x the palm box, shifted half a box toward the fingers.
            let wrist = SIMD2(merged[4], merged[5]), middle = SIMD2(merged[8], merged[9])
            let rotation = atan2(wrist.y - middle.y, wrist.x - middle.x) - .pi / 2
            let centre = SIMD2((merged[0] + merged[2]) / 2, (merged[1] + merged[3]) / 2)
                + Self.rotate(SIMD2(0, -0.5 * boxSize), rotation)
            regions.append(Region(centre: centre, size: 2.6 * boxSize, rotation: rotation))
        }
        return regions
    }

    /// MediaPipe's landmarks -> next-frame region: rotation from the wrist to the middle of the
    /// knuckles, bounding box of palm and finger-base joints in that rotated frame, doubled,
    /// squared, nudged toward the fingers.
    private static func region(fromLandmarks p: [SIMD2<Float>]) -> Region {
        let knuckles = ((p[5] + p[13]) / 2 + p[9]) / 2
        let rotation = atan2(p[0].y - knuckles.y, p[0].x - knuckles.x) - .pi / 2
        var lo = SIMD2<Float>(repeating: .infinity), hi = SIMD2<Float>(repeating: -.infinity)
        for i in [0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18] {
            let local = rotate(p[i] - p[0], -rotation)
            lo = simd_min(lo, local)
            hi = simd_max(hi, local)
        }
        let extent = hi - lo
        let size = max(extent.x, extent.y) * 2
        let centre = p[0] + rotate((lo + hi) / 2 + SIMD2(0, -0.1 * max(extent.x, extent.y)), rotation)
        return Region(centre: centre, size: size, rotation: rotation)
    }

    // MARK: Helpers

    private func loadModels() {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine
        do {
            detector = try MLModel(contentsOf: try Self.compiledModel("HandDetector"), configuration: configuration)
            landmarker = try MLModel(contentsOf: try Self.compiledModel("HandLandmarks"), configuration: configuration)
            input = try MLMultiArray(shape: [1, 3, NSNumber(value: Self.inputSize), NSNumber(value: Self.inputSize)],
                                     dataType: .float32)
            let shape = landmarker?.modelDescription.inputDescriptionsByName["image"]?.multiArrayConstraint?.shape
            let side = shape?.last ?? NSNumber(value: Self.inputSize)
            landmarkInput = try MLMultiArray(shape: [1, 3, side, side], dataType: .float32)
            Log.info("hands: MediaPipe models loaded (Neural Engine), landmarks \(side)x\(side)")
        } catch {
            Log.info("hands: model load failed: \(error.localizedDescription)")
        }
    }

    /// The bundled .mlpackage compiled on the device (there is no Core ML compiler on Linux), cached
    /// in Application Support until the package changes.
    private static func compiledModel(_ name: String) throws -> URL {
        guard let package = Bundle.module.url(forResource: name, withExtension: "mlpackage", subdirectory: "HandModels")
        else { throw CameraError("\(name).mlpackage missing from the app") }
        let weights = package.appendingPathComponent("Data/com.apple.CoreML/weights/weight.bin")
        let stamp = ((try? FileManager.default.attributesOfItem(atPath: weights.path))?[.size] as? Int) ?? 0
        let cache = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("HandModels", isDirectory: true)
        let compiled = cache.appendingPathComponent("\(name)-\(stamp).mlmodelc", isDirectory: true)
        if FileManager.default.fileExists(atPath: compiled.path) { return compiled }
        try FileManager.default.createDirectory(at: cache, withIntermediateDirectories: true)
        let temporary = try MLModel.compileModel(at: package)
        try? FileManager.default.removeItem(at: compiled)
        try FileManager.default.moveItem(at: temporary, to: compiled)
        Log.info("hands: compiled \(name)")
        return compiled
    }

    private static func features(_ input: MLMultiArray) -> MLFeatureProvider {
        // swiftlint:disable:next force_try
        try! MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(multiArray: input)])
    }

    /// Model output as a flat Float array in logical order. Neural Engine outputs are often padded
    /// (strides wider than the shape), so this walks the strides with raw pointers; per-element
    /// MLMultiArray subscripting boxes every value and costs ~10 ms on the palm detector's output.
    private static func read(_ array: MLMultiArray) -> [Float] {
        let shape = array.shape.map(\.intValue), strides = array.strides.map(\.intValue)
        let count = shape.reduce(1, *)
        var out = [Float](repeating: 0, count: count)
        guard count > 0 else { return out }
        let inner = shape.last ?? 1, innerStride = strides.last ?? 1
        let rows = count / inner
        func copy<T>(_ type: T.Type, _ convert: (T) -> Float) {
            let base = array.dataPointer.assumingMemoryBound(to: T.self)
            out.withUnsafeMutableBufferPointer { o in
                for row in 0..<rows {
                    // Offset of this row: decompose the row index over the outer dimensions.
                    var offset = 0, rest = row
                    for d in stride(from: shape.count - 2, through: 0, by: -1) {
                        offset += (rest % shape[d]) * strides[d]
                        rest /= shape[d]
                    }
                    for j in 0..<inner { o[row * inner + j] = convert(base[offset + j * innerStride]) }
                }
            }
        }
        switch array.dataType {
        case .float32: copy(Float.self) { $0 }
        case .float16: copy(Float16.self) { Float($0) }
        default: for i in 0..<count { out[i] = array[i].floatValue }
        }
        return out
    }

    private static func iou(_ a: [Float], _ b: [Float]) -> Float {
        let w = max(0, min(a[2], b[2]) - max(a[0], b[0])), h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        let intersection = w * h
        let union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
        return union > 0 ? intersection / union : 0
    }

    private static func rotate(_ v: SIMD2<Float>, _ angle: Float) -> SIMD2<Float> {
        let c = cos(angle), s = sin(angle)
        return SIMD2(c * v.x - s * v.y, s * v.x + c * v.y)
    }

    /// MediaPipe palm anchors for a 256 input: strides 8, 16, 32, 32, 32 -> 2944 cell centres
    /// (normalised), 2 per cell per layer; same-stride layers share a grid.
    private static func makeAnchors() -> [SIMD2<Float>] {
        let strides = [8, 16, 32, 32, 32]
        var anchors: [SIMD2<Float>] = []
        var layer = 0
        while layer < strides.count {
            var last = layer
            while last < strides.count, strides[last] == strides[layer] { last += 1 }
            let perCell = 2 * (last - layer)
            let grid = inputSize / strides[layer]
            for y in 0..<grid {
                for x in 0..<grid {
                    let centre = SIMD2((Float(x) + 0.5) / Float(grid), (Float(y) + 0.5) / Float(grid))
                    anchors.append(contentsOf: repeatElement(centre, count: perCell))
                }
            }
            layer = last
        }
        return anchors
    }

    /// The locked camera buffer's planes.
    private struct Image {
        let luma: UnsafeMutablePointer<UInt8>
        let lumaStride: Int
        let chroma: UnsafeMutablePointer<UInt8>
        let chromaStride: Int
        let width: Int
        let height: Int

        init?(_ buffer: CVPixelBuffer) {
            guard CVPixelBufferGetPlaneCount(buffer) >= 2,
                  let y = CVPixelBufferGetBaseAddressOfPlane(buffer, 0),
                  let c = CVPixelBufferGetBaseAddressOfPlane(buffer, 1) else { return nil }
            luma = y.assumingMemoryBound(to: UInt8.self)
            chroma = c.assumingMemoryBound(to: UInt8.self)
            lumaStride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 0)
            chromaStride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 1)
            width = CVPixelBufferGetWidthOfPlane(buffer, 0)
            height = CVPixelBufferGetHeightOfPlane(buffer, 0)
        }

        func crop(_ m: [Float], size: Int, into input: MLMultiArray) {
            m.withUnsafeBufferPointer { affine in
                weee_crop_rgb(luma, Int32(lumaStride), chroma, Int32(chromaStride), Int32(width), Int32(height),
                              affine.baseAddress, Int32(size),
                              input.dataPointer.assumingMemoryBound(to: Float.self))
            }
        }
    }
}
