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
    private let handModel = HandModel()              // render thread only

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
    private var handResetPending = false
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
        super.init()
        let onStatus: (String) -> Void = { [weak self] status in self?.setSourceStatus(status) }
        avSource.onStatus = onStatus
        arSource.onStatus = onStatus
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
        lensOffset = mode != .arkit ? AVCameraSource.offsetFromMain(lens: lens) : .zero
        resetPoseStateLocked()
        lock.unlock()
        mailbox.clear()
        stats.reset()
        switch mode {
        // Hand tracking sees every passthrough frame after its pose is attached.
        case .camera:
            arSource.stop()
            xrSource.stop()
            avSource.frameHook = { [hands] frame in hands.submit(frame) }
            avSource.start(lens: lens, preset: preset)
        case .xrslam:
            arSource.stop()
            xrSource.start(settings: slamSettings)
            avSource.frameHook = { [xrSource, hands] frame in
                xrSource.process(&frame)
                hands.submit(frame)
            }
            avSource.start(lens: lens, preset: preset)
        case .arkit:
            avSource.stop()
            xrSource.stop()
            avSource.frameHook = nil
            arSource.onFrame = { [hands] frame in hands.submit(frame) }
            arSource.start()
        }
    }

    // MARK: Controls

    func update(viewer: ViewerProfile) {
        lock.lock()
        let oldSlamSettings = self.viewer.slamSettings
        self.viewer = viewer
        let (newMode, newLens) = Self.source(for: viewer.passthrough)
        let restart = (newMode != mode || newLens != lens) && started
        let retune = !restart && started && newMode == .xrslam && viewer.slamSettings != oldSlamSettings
        if newLens != lens { presetIndex = 0 }
        hands.enabled = viewer.handTracking
        mode = newMode
        lens = newLens
        lock.unlock()
        if restart { startCurrentSource() }
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

    /// Records a Kalibr dataset from the current ultra-wide gyro-3DoF source (the caller switches
    /// to it first, which also frees the raw IMU from XRSLAM). Exposure is capped at 4 ms meanwhile.
    func startCalibrationRecording(onFinish: @escaping (URL?) -> Void) {
        guard !recorder.isRecording else { return }
        avSource.frameHook = { [recorder] frame in recorder.add(frame) }
        avSource.limitExposure(to: 0.004)
        recorder.start { [weak self] directory in
            self?.avSource.frameHook = nil
            self?.avSource.limitExposure(to: nil)
            onFinish(directory)
        }
    }

    var isRecordingCalibration: Bool { recorder.isRecording }
    var calibrationStatus: String? { recorder.status }

    func status() -> Status {
        lock.lock()
        defer { lock.unlock() }
        return Status(
            mode: mode,
            source: sourceStatus,
            tracking: mode == .arkit ? arSource.trackingState : mode == .xrslam ? xrSource.trackingState : nil,
            slamInfo: mode == .xrslam ? xrSource.info : nil,
            handsInfo: viewer.handTracking
                ? String(format: "hands %ld · vision %.1f ms @ %.0f Hz", lastHandCount, hands.stats.ms, hands.stats.hz)
                : nil,
            position: lastPosition,
            depth: lastDepth,
            attachedIntrinsics: mode == .arkit || avSource.usingAttachedIntrinsics,
            cameraFovDeg: SIMD2(2 * atan(lastFrameSize.x / 2 / lastIntrinsics.x),
                                2 * atan(lastFrameSize.y / 2 / lastIntrinsics.y)) * 180 / .pi,
            renderBudgetMS: renderBudget * 1000
        )
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
        let uniforms = frame.map {
            makeUniforms(frame: $0, displayTime: update.targetPresentationTimestamp, size: size, hands: &handData)
        } ?? Uniforms()

        let captureTime = frame?.captureTime, arrivalTime = frame?.arrivalTime
        renderer.draw(to: drawable, frame: frame, uniforms: uniforms, hands: handData, presented: { [stats] presentedTime in
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
                              hands handData: inout [SIMD4<Float>]) -> Uniforms {
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
        let depth = vp.autoDepth ? (frame.proxyDepth ?? vp.fixedDepthM) : vp.fixedDepthM

        // Recenter: place the cube 1 m ahead once there is a valid pose (ARKit: once tracking is normal).
        if recenterPending {
            if frame.pose == nil, qDisplay != nil {
                let unrecentered = Frames.rotationY(yawOffset) * worldFromView
                yawOffset = Frames.yaw(of: unrecentered)
                recenterPending = false
                cubeCenter = head + SIMD3<Float>(Self.cubeOffset)
                cubeVisible = true
            } else if frame.pose != nil, frame.trackingNormal {
                recenterPending = false
                cubeCenter = head + SIMD3<Float>(Frames.rotationY(Frames.yaw(of: worldFromView)) * Self.cubeOffset)
                cubeVisible = true
            }
        }
        let cube = cubeCenter, showCube = cubeVisible && vp.showCube

        // Hands: place new detections in the world with the camera pose at their capture time, on the
        // passthrough's depth sphere so they cover the passthrough hands, then predict to this
        // passthrough frame's capture time so they move with the hands as the passthrough shows them.
        if handResetPending {
            handResetPending = false
            handModel.reset()
        }
        var handFlags: Float = 0
        if vp.handTracking {
            let detection = hands.latestDetection()
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
            handModel.update(detection, pose: detectionPose, head: detectionHead, proxyDepth: depth,
                             flip: vp.flipCamera, handScale: vp.handScale)
            handData = handModel.uniforms(time: frame.captureTime)
            lastHandCount = handModel.visibleCount
            if lastHandCount > 0, vp.handOverlay { handFlags = 8 }
        } else {
            lastHandCount = 0
        }
        lastPosition = viewPosition
        lastDepth = depth
        lastIntrinsics = frame.intrinsics
        lastFrameSize = frame.size
        lock.unlock()

        let hud = renderer.hasHUD
        u.optics = SIMD4(vp.k1, vp.k2, vp.fovScale,
                         (vp.showGrid ? 1 : 0) + (showCube ? 2 : 0) + (hud ? 4 : 0) + handFlags)
        u.setCameraFromWorld(cameraRotation.transpose.float)
        u.camPos = SIMD4(cameraPosition, 0)
        u.setWorldFromView(viewRotation)
        u.head = SIMD4(head, depth)
        u.cube = SIMD4(cube, Self.cubeHalfSize)
        u.hud = SIMD4(-HUDRasterizer.size.x / 2, -HUDRasterizer.size.y - 120, HUDRasterizer.size.x, HUDRasterizer.size.y)
        return u
    }
}
