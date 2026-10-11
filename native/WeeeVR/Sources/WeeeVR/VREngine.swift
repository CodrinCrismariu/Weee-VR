import AVFoundation
import Foundation
import Metal
import QuartzCore
import simd

/// Owns the camera sources, head tracking and the render loop.
///
/// Tracking: XRSLAM visual-inertial odometry on the 0.5× ultra-wide (6DoF, wide passthrough), ARKit
/// world tracking on the 1× (6DoF, Apple's SLAM), or plain camera + gyro (3DoF) for comparison.
/// XRSLAM and ARKit both attach a capture-time world pose to each frame, so the renderer treats
/// them the same.
///
/// Render loop: a CAMetalDisplayLink with preferredFrameLatency = 1 on a dedicated thread. With
/// late-latching, each callback sleeps until just before the frame's commit deadline, then takes the
/// newest camera frame and pose, so frames don't sit waiting for the next refresh.
///
/// Reprojection: the camera image is projected onto a depth-proxy sphere and viewed from each eye's
/// real position at the predicted display time (gyro rotation + ARKit position extrapolated with
/// velocity). That corrects the camera/eye offset and the capture->display head motion, exact at the
/// proxy depth.
final class VREngine: NSObject, CAMetalDisplayLinkDelegate {
    enum Mode: String {
        case camera = "Camera 3DoF"
        case arkit = "ARKit 6DoF"
        case xrslam = "XRSLAM 6DoF"
    }

    struct Status {
        var mode: Mode
        var source: String
        var tracking: String?
        var slamInfo: String?
        var handsInfo: String?
        var gameInfo: String?
        var paddleInfo: String?
        var paddleMessage: String
        var position: SIMD3<Float>
        var depth: Float
        var attachedIntrinsics: Bool
        var cameraFovDeg: SIMD2<Float>   // horizontal, vertical, from the intrinsics actually used
        var renderBudgetMS: Double
    }

    private static let cubeOffset = SIMD3<Double>(0, -0.15, -1.0)   // 1 m ahead, a bit below eye level
    private static let cubeHalfSize: Float = 0.15
    private static let maxSpeed: Float = 2           // m/s cap for position extrapolation
    private static let maxExtrapolation = 0.1        // s

    let stats = LatencyStats()
    let mailbox = FrameMailbox()
    private let renderer: Renderer
    private let motion = MotionTracker()
    private let textures: TextureMaker
    private let avSource: AVCameraSource
    private let arSource: ARKitSource
    private let xrSource: XRSLAMSource
    private let recorder: CalibrationRecorder
    private let hands = HandTracker()
    private let handModel = HandModel()              // render thread only: drawn, on the passthrough sphere
    private let gameHands: HandModel = {             // render thread only: true distance, for the game
        let model = HandModel()
        model.trueDepth = true
        return model
    }()
    private let game = PingPong()                    // render thread only
    private let gameAudio = GameAudio()
    private let palmGesture = PalmGesture()          // render thread only
    private let paddleLink = PaddleLink()            // the smart paddle's IMU stream and haptics
    private let tags = TagTracker()                  // its AprilTag in the camera frames
    private let paddleFusion = PaddleFusion()        // render thread only
    /// The smart paddle finished its one-time calibration (save it in the profile).
    var onPaddleCalibrated: ((PaddleCalibration) -> Void)?
    /// The small tags' placements were learned (save them in the profile).
    var onPaddleSmallTagsLearned: (([Int: Int]) -> Void)?

    private let lock = NSLock()
    private var viewer: ViewerProfile
    private var mode: Mode
    private var lens: AVCameraSource.Lens
    private var lensOffset = SIMD3<Float>.zero        // passthrough camera relative to the 1× (eye offset reference)
    private var presetIndex = 0
    private var sourceStatus = "starting…"
    private var started = false
    private var displayLink: CAMetalDisplayLink?

    // Render-thread state (guarded by `lock` where read elsewhere)
    private var recenterPending = true
    private var yawOffset = 0.0
    private var cubeCenter = SIMD3<Float>(VREngine.cubeOffset)
    private var cubeVisible = false
    private var positionOffset = SIMD3<Float>.zero   // keeps position frozen / smooth across tracking loss
    private var frozenPosition: SIMD3<Float>?
    private var lastGoodPosition: SIMD3<Float>?
    private var lastRenderTime = 0.0
    private var lastPosition = SIMD3<Float>.zero
    private var lastDepth: Float = 1.5
    private var lastIntrinsics = SIMD4<Float>(1, 1, 0, 0)
    private var lastFrameSize = SIMD2<Float>(0, 0)
    private var lastHandCount = 0
    private var lastGameSummary: String?
    private var handResetPending = false
    private var paddleCalibrationPending = false
    private var paddleRecordPending = false
    private var paddleRecording: (lines: [String], until: Double)?   // render thread: raw paddle data
    private var lastTagSequence: UInt64 = 0
    private var lastHandSequence: UInt64 = 0
    private var lastPaddleSummary: String?
    private var hitBuzz: Float = 0.8                 // render thread: buzz strength for `onPlayerHit`
    private var tagOutline: (corners: [SIMD3<Float>], time: Double)?   // render thread: on the passthrough sphere
    private var tagFlash: (success: Bool, until: Double)?              // render thread: calibration result
    private var workEstimate = 0.002                 // peak-decayed CPU + GPU seconds per frame
    private var missPenalty = 0.0

    init(viewer: ViewerProfile) throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw CameraError("Metal is unavailable") }
        renderer = try Renderer(device: device, pixelFormat: .bgra8Unorm)
        textures = TextureMaker(device: device)
        avSource = AVCameraSource(textures: textures, mailbox: mailbox)
        arSource = ARKitSource(textures: textures, mailbox: mailbox)
        xrSource = XRSLAMSource(motion: motion)
        recorder = CalibrationRecorder(motion: motion)
        self.viewer = viewer
        (mode, lens) = Self.source(for: viewer.passthrough)
        hands.enabled = viewer.handTracking
        hands.flipped = viewer.flipCamera
        super.init()
        let onStatus: (String) -> Void = { [weak self] status in self?.setSourceStatus(status) }
        avSource.onStatus = onStatus
        arSource.onStatus = onStatus
        game.onSound = { [gameAudio] kind, position, speed in gameAudio.play(kind, at: position, speed: speed) }
        game.onEvent = { Log.info("game: \($0)") }
        game.onPlayerHit = { [weak self] speed in self?.buzzForHit(speed: speed) }
        paddleFusion.onEvent = { Log.info($0) }
        paddleFusion.onCalibrated = { [weak self] calibration in
            guard let self else { return }
            tagFlash = (true, CACurrentMediaTime() + 1.5)
            paddleLink.buzz(strength: 0.8, milliseconds: 120)
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.25) { [link = self.paddleLink] in
                link.buzz(strength: 0.8, milliseconds: 120)
            }
            DispatchQueue.main.async { self.onPaddleCalibrated?(calibration) }
        }
        paddleFusion.onCalibrationFailed = { [weak self] _ in
            guard let self else { return }
            tagFlash = (false, CACurrentMediaTime() + 1.5)
            let link = paddleLink
            for i in 0..<3 {                                   // three buzzes: it gave up
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.3 * Double(i)) { link.buzz(strength: 0.8, milliseconds: 150) }
            }
        }
        paddleFusion.onSmallTagsLearned = { [weak self] turns in
            DispatchQueue.main.async { self?.onPaddleSmallTagsLearned?(turns) }
        }
        paddleLink.onMessage = { Log.info("paddle link: \($0)") }
        paddleLink.connect(to: viewer.smartPaddle ? viewer.paddleHost : nil)
        tags.enabled = viewer.smartPaddle && viewer.scene == .pingPong
        tags.flipped = viewer.flipCamera
        gameAudio.volume = viewer.gameVolume
        DispatchQueue.global(qos: .utility).async {
            for line in GameSelfTest.run() { Log.info("game self-test: \(line)") }
        }
    }

    var device: MTLDevice { renderer.device }

    // MARK: Lifecycle

    func attach(to layer: CAMetalLayer) {
        guard displayLink == nil else { return }
        layer.device = renderer.device
        layer.pixelFormat = .bgra8Unorm
        layer.framebufferOnly = true
        layer.maximumDrawableCount = 2       // one on screen, one being drawn: no extra queued frame

        let link = CAMetalDisplayLink(metalLayer: layer)
        link.delegate = self
        link.preferredFrameLatency = 1
        link.preferredFrameRateRange = CAFrameRateRange(minimum: 60, maximum: 60, preferred: 60)
        displayLink = link

        let thread = Thread {
            link.add(to: .current, forMode: .default)
            RunLoop.current.run()
        }
        thread.name = "WeeeVR.render"
        thread.qualityOfService = .userInteractive
        thread.start()
    }

    func start() {
        motion.start()
        AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
            guard let self else { return }
            if granted {
                self.lock.lock()
                self.started = true
                self.lock.unlock()
                self.startCurrentSource()
            } else {
                self.setSourceStatus("camera permission denied: enable it in Settings > Weee VR")
            }
        }
    }

    private static func source(for passthrough: Passthrough) -> (Mode, AVCameraSource.Lens) {
        switch passthrough {
        case .xrslam: (.xrslam, .ultraWide)
        case .arkit: (.arkit, .main)
        case .ultraWide: (.camera, .ultraWide)
        case .main: (.camera, .main)
        }
    }

    private func startCurrentSource() {
        lock.lock()
        let presets = AVCameraSource.presets
        if mode != .arkit, !AVCameraSource.isAvailable(lens: lens, preset: presets[presetIndex]) {
            presetIndex = presets.indices.first { AVCameraSource.isAvailable(lens: lens, preset: presets[$0]) } ?? 0
        }
        let mode = self.mode, lens = self.lens, preset = presets[presetIndex], slamSettings = viewer.slamSettings
        let exposure = viewer.exposureCap
        lensOffset = mode != .arkit ? AVCameraSource.offsetFromMain(lens: lens) : .zero
        resetPoseStateLocked()
        lock.unlock()
        mailbox.clear()
        stats.reset()
        switch mode {
        // Hand tracking (and the paddle's tag) sees every passthrough frame after its pose is attached.
        case .camera:
            arSource.stop()
            xrSource.stop()
            avSource.frameHook = { [hands, tags] frame in
                hands.submit(frame)
                tags.submit(frame)
            }
            avSource.start(lens: lens, preset: preset)
            avSource.capExposure(to: exposure)
        case .xrslam:
            arSource.stop()
            xrSource.start(settings: slamSettings)
            avSource.frameHook = { [xrSource, hands, tags] frame in
                xrSource.process(&frame)
                hands.submit(frame)
                tags.submit(frame)
            }
            avSource.start(lens: lens, preset: preset)
            avSource.capExposure(to: exposure)
        case .arkit:
            avSource.stop()
            xrSource.stop()
            avSource.frameHook = nil
            arSource.onFrame = { [hands, tags] frame in
                hands.submit(frame)
                tags.submit(frame)
            }
            arSource.start()
        }
    }

    // MARK: Controls

    func update(viewer: ViewerProfile) {
        lock.lock()
        let oldSlamSettings = self.viewer.slamSettings
        let oldExposure = self.viewer.exposureCap
        let oldScene = self.viewer.scene, oldTableDrop = self.viewer.tableDropM
        let oldPaddleHost = self.viewer.smartPaddle ? self.viewer.paddleHost : nil
        self.viewer = viewer
        let (newMode, newLens) = Self.source(for: viewer.passthrough)
        let restart = (newMode != mode || newLens != lens) && started
        let retune = !restart && started && newMode == .xrslam && viewer.slamSettings != oldSlamSettings
        let exposureChanged = viewer.exposureCap != oldExposure
        let sceneChanged = viewer.scene != oldScene || viewer.tableDropM != oldTableDrop
        if sceneChanged { recenterPending = true }
        if newLens != lens { presetIndex = 0 }
        hands.enabled = viewer.handTracking
        hands.flipped = viewer.flipCamera
        tags.enabled = viewer.smartPaddle && viewer.scene == .pingPong
        tags.flipped = viewer.flipCamera
        mode = newMode
        lens = newLens
        lock.unlock()
        let paddleHost = viewer.smartPaddle ? viewer.paddleHost : nil
        if paddleHost != oldPaddleHost { paddleLink.connect(to: paddleHost) }
        gameAudio.volume = viewer.gameVolume
        if restart { startCurrentSource() }
        if exposureChanged, !restart, newMode != .arkit { avSource.capExposure(to: viewer.exposureCap) }
        if retune {
            xrSource.configure(viewer.slamSettings)
            resetTracking()
        }
    }

    /// Screen tap: ARKit/XRSLAM discard their map and origin; Camera mode just recenters the heading.
    func resetTracking() {
        lock.lock()
        let mode = self.mode
        resetPoseStateLocked()
        lock.unlock()
        if mode == .arkit { arSource.reset() }
        if mode == .xrslam { xrSource.reset() }
        Log.info("reset: tracking (\(mode.rawValue))")
    }

    func setHUD(_ texture: MTLTexture?) { renderer.setHUD(texture) }

    // MARK: Smart paddle

    /// One-time paddle calibration: turn the paddle slowly with its tag toward the camera (~5 s).
    func calibratePaddle() {
        lock.lock()
        paddleCalibrationPending = true
        lock.unlock()
        paddleLink.buzz(strength: 0.6, milliseconds: 80)
    }

    /// 20 s of the paddle's raw inputs (IMU, tag poses in the camera, camera poses, palm) to
    /// Documents/paddle-rec/, for replaying the fusion offline.
    func recordPaddle() {
        lock.lock()
        paddleRecordPending = true
        lock.unlock()
        paddleLink.buzz(strength: 0.6, milliseconds: 80)
    }

    /// One-time gyro bias measurement on the board (the paddle lying still for ~2 s).
    func calibratePaddleIMU() { paddleLink.calibrateIMU() }

    func testPaddleBuzz() {
        lock.lock()
        let strength = viewer.paddleHaptics
        lock.unlock()
        paddleLink.buzz(strength: Double(max(strength, 0.3)), milliseconds: 150)
    }

    /// Render thread (inside the game step): a player hit, felt in proportion to its speed.
    private func buzzForHit(speed: Double) {
        guard hitBuzz > 0 else { return }
        let k = min(speed / 8, 1)
        paddleLink.buzz(strength: Double(hitBuzz) * max(k, 0.35), milliseconds: Int(40 + 30 * k))
    }

    /// The calibration ring around the big tag (only while calibrating or just after), world frame:
    /// scene[overlayBase + i] = the tag's corner i, with w of the first = 1 (shown) and w of the second
    /// = ring (0...1 calibrating, 2 done, 3 failed).
    private func writeTagOverlay(into scene: inout [SIMD4<Float>], time: Double) {
        let base = PingPong.overlayBase
        guard let outline = tagOutline, time - outline.time < 1 else { return }
        var ring: Float = -1
        if let p = paddleFusion.calibrationProgress { ring = Float(p) }
        if let flash = tagFlash {
            if CACurrentMediaTime() < flash.until { ring = flash.success ? 2 : 3 } else { tagFlash = nil }
        }
        guard ring >= 0 else { return }
        let alpha: Float = 1
        for i in 0..<4 { scene[base + i] = SIMD4(outline.corners[i], 0) }
        scene[base].w = alpha
        scene[base + 1].w = ring
    }

    /// The camera's world pose at a detection's capture time: XRSLAM/ARKit's pose (plus the render
    /// thread's position offset, as the hands get it), else the gyro attitude at the origin.
    private func detectionPose(_ pose: simd_float4x4?, captureTime: Double) -> (simd_float3x3, SIMD3<Float>)? {
        if let pose { return (pose.rotation.float, pose.translation + positionOffset) }
        guard let q = motion.attitude(at: captureTime) else { return nil }
        let r = Frames.rotationY(-yawOffset) * Frames.zUpToYUp * simd_double3x3(q) * Frames.viewToDevice
        return (r.float, .zero)
    }

    /// Render thread: new IMU samples, tag sightings and right-palm positions into the fusion.
    private func feedPaddle(_ vp: ViewerProfile, handDetection: HandTracker.Detection?, time: Double, proxyDepth: Float) {
        if paddleFusion.calibration != vp.paddleCalibration { paddleFusion.calibration = vp.paddleCalibration }
        if paddleFusion.smallTurns != vp.paddleSmallTagTurns { paddleFusion.smallTurns = vp.paddleSmallTagTurns }
        paddleFusion.tagTurnsOverride = vp.paddleTagTurns >= 0 ? vp.paddleTagTurns : nil
        if paddleCalibrationPending {
            paddleCalibrationPending = false
            paddleFusion.beginCalibration()
        }
        paddleFusion.tick(time: time)
        if paddleRecordPending {
            paddleRecordPending = false
            let c = vp.paddleCalibration
            paddleRecording = (["C," + (c.map { "\($0.mount.map { String($0) }.joined(separator: ",")),\($0.timeOffset),\($0.tagTurns)" } ?? "none")],
                               time + 20)
        }
        let samples = paddleLink.take()
        if paddleRecording != nil {
            for s in samples {
                paddleRecording!.lines.append(String(format: "I,%.6f,%.6f,%.6f,%.6f,%.6f,%.5f,%.5f,%.5f,%d", s.time,
                                                     s.orientation[0], s.orientation[1], s.orientation[2], s.orientation[3],
                                                     s.rate.x, s.rate.y, s.rate.z, s.still ? 1 : 0))
            }
        }
        if !samples.isEmpty {
            paddleFusion.add(samples.map {
                PaddleIMUSample(time: $0.time,
                                orientation: Quat(w: $0.orientation[0], x: $0.orientation[1], y: $0.orientation[2],
                                                  z: $0.orientation[3]),
                                rate: V3($0.rate.x, $0.rate.y, $0.rate.z), still: $0.still)
            })
        }
        if let detection = tags.latestDetection(), detection.sequence != lastTagSequence {
            lastTagSequence = detection.sequence
            if let (rotation, position) = detectionPose(detection.pose, captureTime: detection.captureTime) {
                let r = simd_double3x3(columns: (SIMD3<Double>(rotation.columns.0), SIMD3<Double>(rotation.columns.1),
                                                  SIMD3<Double>(rotation.columns.2)))
                let p = SIMD3<Double>(position)
                for tag in detection.tags {
                    if paddleRecording != nil {
                        // ID, camera pose (world <- view, columns), each solution (view <- tag, columns + t + error).
                        var line = String(format: "T,%.6f,%d", detection.captureTime, tag.id)
                        for c in [rotation.columns.0, rotation.columns.1, rotation.columns.2, position] {
                            line += String(format: ",%.6f,%.6f,%.6f", c.x, c.y, c.z)
                        }
                        line += ",\(tag.solutions.count)"
                        for s in tag.solutions {
                            for c in [s.rotation.columns.0, s.rotation.columns.1, s.rotation.columns.2, s.translation] {
                                line += String(format: ",%.6f,%.6f,%.6f", c.x, c.y, c.z)
                            }
                            line += String(format: ",%.3e", s.error)
                        }
                        paddleRecording!.lines.append(line)
                    }
                    let candidates = tag.solutions.map { s -> PaddleTagSighting.Candidate in
                        let world = r * s.rotation
                        let centre = p + r * s.translation
                        return PaddleTagSighting.Candidate(
                            rotation: Quat(basis: Basis(x: V3(world.columns.0.x, world.columns.0.y, world.columns.0.z),
                                                        y: V3(world.columns.1.x, world.columns.1.y, world.columns.1.z),
                                                        z: V3(world.columns.2.x, world.columns.2.y, world.columns.2.z))),
                            position: V3(centre.x, centre.y, centre.z), error: s.error)
                    }
                    paddleFusion.add(PaddleTagSighting(id: tag.id, time: detection.captureTime, candidates: candidates))
                    // Where the calibration ring goes: around the big tag, as the passthrough shows it (on
                    // the sphere seen from the camera at capture, as the hands are drawn).
                    if PaddleFusion.bigTagIDs.contains(tag.id), let s = tag.solutions.first {
                        let h = tag.size / 2
                        let head = position + rotation * (vp.eyeOffset - lensOffset)
                        let corners = [SIMD3<Double>(-h, h, 0), SIMD3(h, h, 0), SIMD3(h, -h, 0), SIMD3(-h, -h, 0)].map { c in
                            let world = SIMD3<Float>(p + r * (s.rotation * c + s.translation))
                            let d = simd_normalize(world - position)
                            let oc = position - head
                            let b = simd_dot(oc, d), cc = simd_dot(oc, oc) - proxyDepth * proxyDepth
                            let t = -b + max(b * b - cc, 0).squareRoot()
                            return position + d * t
                        }
                        tagOutline = (corners, detection.captureTime)
                    }
                }
            }
        }
        // The right palm at its true distance, once per hand detection.
        if let handDetection, handDetection.sequence != lastHandSequence {
            lastHandSequence = handDetection.sequence
            if let j = gameHands.joints(chirality: 1), j.count >= 21 {
                let palm = (j[0] + j[5] + j[9] + j[13] + j[17]) / 5
                paddleRecording?.lines.append(String(format: "P,%.6f,%.5f,%.5f,%.5f", handDetection.captureTime,
                                                     palm.x, palm.y, palm.z))
                paddleFusion.addPalm(V3(Double(palm.x), Double(palm.y), Double(palm.z)),
                                     time: handDetection.captureTime)
            }
        }
        finishPaddleRecording(time: time)
    }

    /// Render thread: a finished paddle recording goes to a file.
    private func finishPaddleRecording(time: Double) {
        guard let recording = paddleRecording, time > recording.until else { return }
        paddleRecording = nil
        let link = paddleLink
        DispatchQueue.global(qos: .utility).async {
            guard let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else { return }
            let dir = docs.appendingPathComponent("paddle-rec")
            try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            let name = "rec-\(Int(Date().timeIntervalSince1970)).csv"
            try? recording.lines.joined(separator: "\n").write(to: dir.appendingPathComponent(name), atomically: true, encoding: .utf8)
            Log.info("paddle: recording saved, \(name), \(recording.lines.count) lines")
            link.buzz(strength: 0.8, milliseconds: 120)
            DispatchQueue.global().asyncAfter(deadline: .now() + 0.25) { link.buzz(strength: 0.8, milliseconds: 120) }
        }
    }

    /// Records a Kalibr dataset from the current ultra-wide gyro-3DoF source (the caller switches
    /// to it first, which also frees the raw IMU from XRSLAM). Exposure is capped at 4 ms meanwhile.
    func startCalibrationRecording(onFinish: @escaping (URL?) -> Void) {
        guard !recorder.isRecording else { return }
        avSource.frameHook = { [recorder] frame in recorder.add(frame) }
        avSource.limitExposure(to: 0.004)
        recorder.start { [weak self, hands] directory in
            // Back to the gyro mode's hook (the caller then restores its previous mode, if different).
            self?.avSource.frameHook = { frame in hands.submit(frame) }
            self?.avSource.limitExposure(to: nil)
            onFinish(directory)
        }
    }

    var isRecordingCalibration: Bool { recorder.isRecording }
    var calibrationStatus: String? { recorder.status }

    /// HUD/log line: hands shown, tracker time and rate, lost hands regained / dropped since launch.
    private func handsInfo(shown: Int) -> String {
        let s = hands.stats
        let ms = (s.ms * 10).rounded() / 10
        let n = s.newHands
        return "hands \(shown) · \(ms) ms @ \(Int(s.hz.rounded())) Hz · regained \(s.regained) dropped \(s.dropped)"
            + " · new \(n.started)→\(n.confirmed) palm miss \(n.palmMisses)/\(n.palmRuns)"
            + " rej p\(n.rejectPresence) r\(n.rejectReach) t\(n.rejectTilt)"
    }

    func status() -> Status {
        // Own state under `lock`; the sources' getters outside it, so a source holding its own lock
        // can never make the render thread wait on `lock`.
        lock.lock()
        let mode = self.mode, handTracking = viewer.handTracking, handCount = lastHandCount
        let smartPaddle = viewer.smartPaddle, paddleSummary = lastPaddleSummary
        var status = Status(
            mode: mode,
            source: sourceStatus,
            tracking: nil,
            slamInfo: nil,
            handsInfo: nil,
            gameInfo: lastGameSummary,
            paddleInfo: nil,
            paddleMessage: "off",
            position: lastPosition,
            depth: lastDepth,
            attachedIntrinsics: mode == .arkit,
            cameraFovDeg: SIMD2(2 * atan(lastFrameSize.x / 2 / lastIntrinsics.x),
                                2 * atan(lastFrameSize.y / 2 / lastIntrinsics.y)) * 180 / .pi,
            renderBudgetMS: renderBudget * 1000
        )
        lock.unlock()
        status.tracking = mode == .arkit ? arSource.trackingState : mode == .xrslam ? xrSource.trackingState : nil
        status.slamInfo = mode == .xrslam ? xrSource.info : nil
        status.handsInfo = handTracking ? handsInfo(shown: handCount) : nil
        status.attachedIntrinsics = status.attachedIntrinsics || avSource.usingAttachedIntrinsics
        if smartPaddle {
            let link = paddleLink.currentStats, tag = tags.stats
            let imu = link.streaming
                ? String(format: "imu %.0f Hz age %.0f ms loss %.0f%% rtt %.0f ms%@", link.hz, link.ageMS,
                         link.lossPercent, link.rttMS, link.imuCalibrated == false ? " UNCALIBRATED" : "")
                : (link.connected ? "imu not streaming" : "not connected")
            status.paddleInfo = "paddle " + imu
                + String(format: " · tag %.0f/%.0f Hz %.1f ms", tag.foundHz, tag.hz, tag.ms)
                + (paddleSummary.map { " · " + $0 } ?? "")
            status.paddleMessage = link.message
        }
        return status
    }

    private func setSourceStatus(_ status: String) {
        lock.lock()
        sourceStatus = status
        lock.unlock()
        Log.info(String(format: "source: %@", status))
    }

    private func resetPoseStateLocked() {
        recenterPending = true
        cubeVisible = false
        positionOffset = .zero
        frozenPosition = nil
        lastGoodPosition = nil
        handResetPending = true
    }

    // MARK: CAMetalDisplayLinkDelegate

    /// Time reserved before the commit deadline: measured work + margin, widened after a missed refresh.
    private var renderBudget: Double {
        min(max(workEstimate + 0.0025, 0.003), 0.010) + missPenalty
    }

    func metalDisplayLink(_ link: CAMetalDisplayLink, needsUpdate update: CAMetalDisplayLink.Update) {
        lock.lock()
        let lateLatch = viewer.lateLatch
        lock.unlock()

        let misses = stats.takeNewMisses()
        missPenalty = misses > 0 ? min(missPenalty + 0.001 * Double(misses), 0.006) : missPenalty * 0.995
        if lateLatch {
            let wait = update.targetTimestamp - renderBudget - CACurrentMediaTime()
            if wait > 0.0003 { Thread.sleep(forTimeInterval: min(wait, 0.0167)) }
        }

        let start = CACurrentMediaTime()
        let drawable = update.drawable
        let size = SIMD2<Float>(Float(drawable.texture.width), Float(drawable.texture.height))
        let frame = mailbox.take()
        var handData = HandModel.empty
        var sceneData = [SIMD4<Float>](repeating: .zero, count: PingPong.sceneCount)
        let uniforms = frame.map {
            makeUniforms(frame: $0, displayTime: update.targetPresentationTimestamp, size: size, hands: &handData,
                         scene: &sceneData)
        } ?? Uniforms()

        let captureTime = frame?.captureTime, arrivalTime = frame?.arrivalTime
        renderer.draw(to: drawable, frame: frame, uniforms: uniforms, hands: handData, scene: sceneData,
                      presented: { [stats] presentedTime in
            guard let captureTime, let arrivalTime else { return }
            stats.addPresent(captureTime: captureTime, arrivalTime: arrivalTime, presentedTime: presentedTime)
        }, completed: { [weak self] gpuSeconds in
            self?.stats.addGpuTime(seconds: gpuSeconds)
        })

        let cpu = CACurrentMediaTime() - start
        stats.addRenderCost(cpuSeconds: cpu)
        let gpu = stats.snapshotGpuEstimate
        workEstimate = max(cpu + gpu, workEstimate * 0.98)
    }

    // MARK: Per-frame maths

    private func makeUniforms(frame: CameraFrame, displayTime: Double, size: SIMD2<Float>,
                              hands handData: inout [SIMD4<Float>], scene sceneData: inout [SIMD4<Float>]) -> Uniforms {
        lock.lock()
        let vp = viewer
        lock.unlock()

        var u = Uniforms()
        let centres = vp.lensCenters(drawableWidth: size.x, height: size.y)
        u.screen = SIMD4(size.x, size.y, vp.pixelsPerMeter, vp.screenToLensMM / 1000)
        u.lens = SIMD4(centres.left.x, centres.left.y, centres.right.x, tan(vp.lensFovDeg * .pi / 180))
        u.camK = frame.intrinsics
        u.cam = SIMD4(frame.size.x, frame.size.y, vp.ipdMM / 1000, vp.flipCamera ? 1 : 0)

        let viewToDevice = Frames.viewToDevice
        let displayTarget = displayTime + Double(vp.predictionMS) / 1000
        let qCapture = motion.attitude(at: frame.captureTime)
        let qDisplay = motion.attitude(at: displayTarget)

        // Head rotation from capture to display (view frame), from the gyro.
        var delta = matrix_identity_double3x3
        if vp.reprojection, let qCapture, let qDisplay {
            delta = viewToDevice.transpose * simd_double3x3(qCapture).transpose * simd_double3x3(qDisplay) * viewToDevice
        }

        let now = CACurrentMediaTime()
        let frameDt = Float(min(max(now - lastRenderTime, 0), 0.1))
        lastRenderTime = now

        // Camera pose at capture (passthrough source) and view pose at display time, both in world.
        let cameraRotation: simd_double3x3
        var cameraPosition = SIMD3<Float>.zero
        var worldFromView: simd_double3x3
        var viewPosition = SIMD3<Float>.zero

        lock.lock()
        if let pose = frame.pose {
            // ARKit 6DoF. While tracking is limited the position holds still (rotation keeps working);
            // when it recovers, the offset decays so the position glides back instead of jumping.
            let raw = pose.translation
            if frame.trackingNormal {
                frozenPosition = nil
                positionOffset *= exp(-frameDt / 0.3)
            } else if let lastGood = lastGoodPosition {
                if frozenPosition == nil { frozenPosition = lastGood }
                positionOffset = frozenPosition! - raw
            }
            cameraRotation = pose.rotation
            cameraPosition = raw + positionOffset
            if frame.trackingNormal { lastGoodPosition = cameraPosition }
            worldFromView = cameraRotation * delta
            viewPosition = cameraPosition
            if vp.reprojection, frame.trackingNormal {
                let dt = Float(min(max(displayTarget - frame.captureTime, 0), Self.maxExtrapolation))
                var v = frame.velocity
                let speed = simd_length(v)
                if speed > Self.maxSpeed { v *= Self.maxSpeed / speed }
                viewPosition += v * dt
            }
        } else {
            // Camera mode: gyro only (3DoF), heading recentered by `yawOffset`.
            let identity = simd_quatd(ix: 0, iy: 0, iz: 0, r: 1)
            let recenter = Frames.rotationY(-yawOffset)
            cameraRotation = recenter * Frames.zUpToYUp * simd_double3x3(qCapture ?? identity) * viewToDevice
            worldFromView = vp.reprojection
                ? recenter * Frames.zUpToYUp * simd_double3x3(qDisplay ?? identity) * viewToDevice
                : cameraRotation
        }

        // Eye midpoint: offset from the passthrough camera, rotated with the head.
        let viewRotation = worldFromView.float
        let head = viewPosition + viewRotation * (vp.eyeOffset - lensOffset)
        let proxyDepth = vp.autoDepth ? (frame.proxyDepth ?? vp.fixedDepthM) : vp.fixedDepthM

        // Recenter: place the cube 1 m ahead / the table in front once there is a valid pose (ARKit,
        // XRSLAM: once tracking is normal).
        let playing = vp.scene == .pingPong
        if recenterPending {
            if frame.pose == nil, qDisplay != nil {
                let unrecentered = Frames.rotationY(yawOffset) * worldFromView
                yawOffset = Frames.yaw(of: unrecentered)
                recenterPending = false
                cubeCenter = head + SIMD3<Float>(Self.cubeOffset)
                cubeVisible = true
                if playing { game.place(head: head, yaw: 0) }      // heading recentered to 0 from now on
            } else if frame.pose != nil, frame.trackingNormal {
                recenterPending = false
                cubeCenter = head + SIMD3<Float>(Frames.rotationY(Frames.yaw(of: worldFromView)) * Self.cubeOffset)
                cubeVisible = true
                if playing { game.place(head: head, yaw: Frames.yaw(of: worldFromView)) }
            }
        }
        let cube = cubeCenter, showCube = cubeVisible && vp.scene == .cube

        // Hands: place new detections in the world with the camera pose at their capture time, on the
        // passthrough's depth sphere so they cover the passthrough hands, predicted to this passthrough
        // frame's capture time so they move with the hands as the passthrough shows them. The game also
        // keeps them at their true distance (`gameHands`, not drawn) to hit and pick up the ball.
        if handResetPending {
            handResetPending = false
            handModel.reset()
            gameHands.reset()
        }
        handModel.lostHold = playing ? 0.3 : 0.12
        gameHands.lostHold = handModel.lostHold
        var handFlags: Float = 0
        hitBuzz = vp.paddleHaptics
        var handDetection: HandTracker.Detection?
        if vp.handTracking {
            let detection = hands.latestDetection()
            handDetection = detection
            var detectionPose: (simd_float3x3, SIMD3<Float>)?
            if let detection {
                if let pose = detection.pose {
                    detectionPose = (pose.rotation.float, pose.translation + positionOffset)
                } else if let q = motion.attitude(at: detection.captureTime) {
                    let r = Frames.rotationY(-yawOffset) * Frames.zUpToYUp * simd_double3x3(q) * viewToDevice
                    detectionPose = (r.float, .zero)
                }
            }
            // The passthrough's depth sphere as it was at the detection's capture time.
            let detectionHead = detectionPose.map { $0.1 + $0.0 * (vp.eyeOffset - lensOffset) } ?? head
            handModel.update(detection, pose: detectionPose, head: detectionHead, proxyDepth: proxyDepth,
                             flip: vp.flipCamera, handScale: vp.handScale)
            handModel.highlight[0] = palmGesture.progress
            handData = handModel.uniforms(time: frame.captureTime + Double(vp.paddlePrediction), maxLead: 0.12)
            if playing {
                gameHands.update(detection, pose: detectionPose, head: detectionHead, proxyDepth: proxyDepth,
                                 flip: vp.flipCamera, handScale: vp.handScale)
                _ = gameHands.uniforms(time: frame.captureTime + Double(vp.paddlePrediction), maxLead: 0.12)
            }
            // In-headset recenter: open left palm toward the face for a second.
            if palmGesture.update(time: displayTarget, joints: handModel.joints(chirality: 0), head: head) {
                recenterPending = true
                Log.info("gesture: recenter")
            }
            lastHandCount = handModel.visibleCount
            if lastHandCount > 0, vp.handOverlay { handFlags = 8 }
        } else {
            lastHandCount = 0
        }

        // The game, stepped to the display time with the hands at their true distance; the paddle is
        // drawn where it is (true distance), the hands where the camera shows them (the sphere).
        var sceneFlags: Float = 0
        if playing, game.placed {
            game.settings = GameSettings(tableDrop: Double(vp.tableDropM), opponentLevel: Double(vp.opponentLevel),
                                         gripAngle: Double(vp.gripAngleDeg) * .pi / 180,
                                         gripOffset: Double(vp.gripOffsetMM) / 1000,
                                         assist: Double(vp.assist), floatingServe: vp.floatingServe)
            let left = vp.handTracking ? gameHands.joints(chirality: 0) : nil
            if vp.smartPaddle { feedPaddle(vp, handDetection: handDetection, time: frame.captureTime, proxyDepth: proxyDepth) }
            if vp.smartPaddle, paddleFusion.pose(at: displayTarget) != nil {
                // The smart paddle, sampled along its measured path at every physics substep.
                game.update(time: displayTarget, path: { [paddleFusion] t in paddleFusion.pose(at: t) ?? PaddleState() },
                            left: left)
            } else {
                game.update(time: displayTarget, right: vp.handTracking ? gameHands.joints(chirality: 1) : nil, left: left)
            }
            lastPaddleSummary = vp.smartPaddle ? paddleFusion.summary : nil
            // Tell the tag search where the IMU says the tag is now (the next frames are captured
            // about now), in the camera poses' own coordinates.
            if vp.smartPaddle, paddleFusion.calibration != nil {
                let now = CACurrentMediaTime()
                if let c = paddleFusion.predictedCentre(at: now) {
                    tags.setHint(centre: SIMD3<Float>(Float(c.x), Float(c.y), Float(c.z)) - positionOffset, radius: 0.12,
                                 time: now)
                }
            }
            sceneData = game.sceneData()
            if vp.smartPaddle { writeTagOverlay(into: &sceneData, time: frame.captureTime) }
            sceneFlags = 16
            gameAudio.setListener(position: head, forward: viewRotation * SIMD3<Float>(0, 0, -1),
                                  up: viewRotation * SIMD3<Float>(0, 1, 0))
            lastGameSummary = game.summary
        } else {
            lastGameSummary = nil
        }
        lastPosition = viewPosition
        lastDepth = proxyDepth
        lastIntrinsics = frame.intrinsics
        lastFrameSize = frame.size
        lock.unlock()

        let hud = renderer.hasHUD
        u.optics = SIMD4(vp.k1, vp.k2, vp.fovScale,
                         (vp.showGrid ? 1 : 0) + (showCube ? 2 : 0) + (hud ? 4 : 0)
                         + handFlags + sceneFlags)
        u.setCameraFromWorld(cameraRotation.transpose.float)
        u.camPos = SIMD4(cameraPosition, 0)
        u.setWorldFromView(viewRotation)
        u.head = SIMD4(head, proxyDepth)
        u.cube = SIMD4(cube, Self.cubeHalfSize)
        u.hud = SIMD4(-HUDRasterizer.size.x / 2, -HUDRasterizer.size.y - 120, HUDRasterizer.size.x, HUDRasterizer.size.y)
        u.image = SIMD4(vp.accurateColor ? frame.ycbcrMatrix : 0, vp.sharpening, vp.blackLift, vp.edgeSoftness)
        u.chroma = SIMD4(-vp.chromaticAberration, vp.chromaticAberration,
                         vp.accurateColor && frame.needsP3Conversion ? 1 : 0, 0)
        return u
    }
}
