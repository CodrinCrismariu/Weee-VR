import Accelerate
import CoreVideo
import Foundation
import QuartzCore
import XRSLAMCore
import simd

/// 6DoF on the ultra-wide: XRSLAM visual-inertial odometry on the same camera feed as the passthrough.
///
/// - Tracking input (serial `queue`): every n-th camera frame's Y plane scaled to 640×480, plus raw
///   gyro/accelerometer at 100 Hz. XRSLAM runs its feature tracker and bundle adjustment on its own
///   threads below the render thread's priority.
/// - Pose output (`process`, on the capture queue, every frame): XRSLAM's latest pose (1–3 frames
///   old) advanced to the frame's capture time with the gyro and velocity, then handed to the
///   renderer exactly like an ARKit frame pose.
/// - Continuity: when XRSLAM loses tracking it starts a new map with a new origin and heading. The
///   new map is re-anchored onto the old one (heading from the gyro, position from the last good
///   one), so the world doesn't jump. Only a tap (`reset`) moves the origin.
final class XRSLAMSource {
    struct Settings: Equatable {
        var timeOffsetMS: Float = 0
        var leverArmMM = SIMD3<Float>.zero      // camera position in the IMU (device) frame
        var feedHz = 30
        var keypoints = 300
        var window = 5
    }

    static let width = 640, height = 480
    private static let flipYZ = simd_double3x3(diagonal: SIMD3(1, -1, -1))   // OpenCV camera <-> view frame
    private static let maxStaleness = 0.3                                    // s; older poses = not tracking
    private static let headingTimeConstant = 1.0     // s: how fast XRSLAM's heading pulls the gyro's
    private static let positionTimeConstant = 0.12   // s: low-pass on XRSLAM position corrections
    private static let positionSnapDistance = 0.5    // m: larger corrections are applied at once

    private let motion: MotionTracker
    private let queue = DispatchQueue(label: "xrslam.feed", qos: .userInitiated)
    private let imuQueue = OperationQueue()

    // Guarded by `lock`. `slam` is only created/destroyed on `queue`.
    private let lock = NSLock()
    private var slam: OpaquePointer?
    private var settings = Settings()
    private var feedBusy = false
    private var resetPending = false
    private var _trackingState = "—"
    private var _info = ""

    // `queue` only
    private var scaled = [UInt8](repeating: 0, count: XRSLAMSource.width * XRSLAMSource.height)
    private var fedImages = 0

    // Capture queue only (`process`)
    private var frameCount = 0
    private var droppedFeeds = 0
    private var wasTracking = false
    private var hadSession = false
    private var anchorRotation = matrix_identity_double3x3
    private var anchorTranslation = SIMD3<Double>.zero
    private var align = matrix_identity_double3x3        // our world <- gyro world (heading only)
    private var alignYaw = 0.0
    private var filteredPosition: SIMD3<Double>?
    private var lastFilterTime = 0.0
    private var lastGoodPosition: SIMD3<Double>?
    private var depth: Float?
    private var lastDepthTime = 0.0

    init(motion: MotionTracker) {
        self.motion = motion
        imuQueue.underlyingQueue = queue
        imuQueue.maxConcurrentOperationCount = 1
    }

    var trackingState: String {
        lock.lock()
        defer { lock.unlock() }
        return _trackingState
    }

    /// One-line tracker diagnostics for the HUD and log.
    var info: String {
        lock.lock()
        defer { lock.unlock() }
        return _info
    }

    func start(settings: Settings) {
        configure(settings)
        motion.startRawIMU(interval: 0.01, queue: imuQueue, gyro: { [weak self] t, w in
            guard let slam = self?.slam else { return }
            weee_slam_push_gyro(slam, t, w.x, w.y, w.z)
        }, accel: { [weak self] t, a in
            guard let slam = self?.slam else { return }
            let g = a * -9.80665                    // CoreMotion g (gravity reads -1) -> m/s², +g up
            weee_slam_push_accel(slam, t, g.x, g.y, g.z)
        })
    }

    func stop() {
        motion.stopRawIMU()
        queue.async { [self] in destroy() }
    }

    /// New calibration or tracker size: the tracker is rebuilt on the next frame.
    func configure(_ settings: Settings) {
        queue.async { [self] in
            lock.lock()
            self.settings = settings
            lock.unlock()
            destroy()
        }
        lock.lock()
        resetPending = true
        lock.unlock()
    }

    /// Tap: drop the map and move the origin to the current position.
    func reset() {
        queue.async { [self] in
            if let slam { weee_slam_reset(slam) }
            fedImages = 0
        }
        lock.lock()
        resetPending = true
        lock.unlock()
        Log.info("reset: XRSLAM map and position")
    }

    // MARK: Capture queue

    /// Attach the 6DoF pose at capture time to `frame`, and feed every n-th frame to the tracker.
    func process(_ frame: inout CameraFrame) {
        let t = frame.captureTime
        lock.lock()
        let feedEvery = max(1, 60 / max(settings.feedHz, 1))
        if resetPending {
            resetPending = false
            hadSession = false
            wasTracking = false
            anchorRotation = matrix_identity_double3x3
            anchorTranslation = .zero
            lastGoodPosition = nil
            filteredPosition = nil
            lastFilterTime = 0
            depth = nil
        }
        var state = WeeeSLAMState()
        if let slam { weee_slam_get_state(slam, &state) }
        let hasTracker = slam != nil
        lock.unlock()

        frameCount += 1
        if frameCount % feedEvery == 0 { feed(frame) }

        let now = CACurrentMediaTime()
        let lag = now - state.t
        let attitudeNow = motion.attitude(at: t)
        let gyroView = attitudeNow.map { Frames.zUpToYUp * simd_double3x3($0) * Frames.viewToDevice }
        let tracking = state.status == Int32(WEEE_SLAM_TRACKING.rawValue) && state.has_pose != 0
            && lag < Self.maxStaleness

        var rotation: simd_double3x3
        var position: SIMD3<Double>
        var velocity = SIMD3<Double>.zero
        if tracking {
            let q = simd_quatd(ix: state.q_wc.0, iy: state.q_wc.1, iz: state.q_wc.2, r: state.q_wc.3)
            rotation = Frames.zUpToYUp * simd_double3x3(q) * Self.flipYZ
            position = Frames.zUpToYUp * SIMD3(state.p_wc.0, state.p_wc.1, state.p_wc.2)
            velocity = Frames.zUpToYUp * SIMD3(state.v_wb.0, state.v_wb.1, state.v_wb.2)
            let attitudeAtPose = motion.attitude(at: state.t)

            if !wasTracking {
                // A new XRSLAM map. After a tap it defines the world; after a tracking loss it is
                // anchored so this pose continues where the old map left off.
                if hadSession, let lastGood = lastGoodPosition, let a = attitudeAtPose {
                    let expected = align * Frames.zUpToYUp * simd_double3x3(a) * Frames.viewToDevice
                    anchorRotation = Self.headingOnly(expected * rotation.transpose)
                    anchorTranslation = lastGood - anchorRotation * position
                    Log.info("xrslam: tracking regained, map re-anchored")
                } else {
                    anchorRotation = matrix_identity_double3x3
                    anchorTranslation = .zero
                    Log.info("xrslam: tracking")
                }
                hadSession = true
            }
            rotation = anchorRotation * rotation
            position = anchorRotation * position + anchorTranslation
            velocity = anchorRotation * velocity

            // Advance the pose from its own time to this frame's capture time.
            if let a = attitudeAtPose, let b = attitudeNow {
                rotation = rotation * Frames.viewToDevice.transpose * simd_double3x3(a).transpose
                    * simd_double3x3(b) * Frames.viewToDevice
            }
            position += velocity * min(max(t - state.t, -0.1), 0.1)

            // Smoothing. XRSLAM's pose steps whenever bundle adjustment corrects it, so rotation comes
            // from the gyro (smooth at 200 Hz; tilt is gravity-referenced) with XRSLAM only steering
            // its heading, and position is XRSLAM's low-passed around its own velocity.
            let dt = lastFilterTime > 0 ? min(max(t - lastFilterTime, 0), 0.1) : 0
            lastFilterTime = t
            if let gyroView {
                let target = Frames.yaw(of: rotation * gyroView.transpose)
                if !wasTracking {
                    alignYaw = target
                } else {
                    let error = remainder(target - alignYaw, 2 * .pi)
                    alignYaw += error * (1 - exp(-dt / Self.headingTimeConstant))
                }
                align = Frames.rotationY(alignYaw)
                rotation = align * gyroView
            }
            if !wasTracking || filteredPosition == nil
                || simd_length(position - filteredPosition!) > Self.positionSnapDistance {
                filteredPosition = position
            } else {
                let predicted = filteredPosition! + velocity * dt
                filteredPosition = predicted + (position - predicted) * (1 - exp(-dt / Self.positionTimeConstant))
            }
            position = filteredPosition!
            lastGoodPosition = position

            if state.depth_median > 0 {
                let measured = Float(state.depth_median)
                let alpha = depth == nil ? 1 : 1 - exp(-Float(max(t - lastDepthTime, 0)) / 0.3)
                depth = min(max((depth ?? measured) + alpha * (measured - (depth ?? measured)), 0.4), 6)
                lastDepthTime = t
            }
        } else {
            // No pose: rotation from the gyro (in this map's heading), position held.
            rotation = align * (gyroView ?? matrix_identity_double3x3)
            position = lastGoodPosition ?? .zero
        }
        wasTracking = tracking

        var pose = matrix_identity_float4x4
        let r = rotation.float
        pose.columns.0 = SIMD4(r.columns.0, 0)
        pose.columns.1 = SIMD4(r.columns.1, 0)
        pose.columns.2 = SIMD4(r.columns.2, 0)
        pose.columns.3 = SIMD4(SIMD3<Float>(position), 1)
        frame.pose = pose
        frame.velocity = SIMD3<Float>(velocity)
        frame.trackingNormal = tracking
        frame.proxyDepth = depth

        let status: String
        if !hasTracker {
            status = "XRSLAM starting…"
        } else if tracking {
            status = "tracking"
        } else if state.status == Int32(WEEE_SLAM_INITIALIZING.rawValue) {
            status = "move side to side to start"
        } else {
            status = "lost"
        }
        // Plain interpolation: this runs every camera frame, and a String(format:) type mismatch makes
        // iOS log an expensive fault each time.
        let lagMS = tracking ? Int((lag * 1000).rounded()) : 0
        let info = "slam lag \(lagMS) ms · \(state.landmarks) lm · \(state.images) img · drop \(droppedFeeds)"
        lock.lock()
        _trackingState = status
        _info = info
        lock.unlock()
    }

    private func feed(_ frame: CameraFrame) {
        lock.lock()
        if feedBusy {
            lock.unlock()
            droppedFeeds += 1
            return
        }
        feedBusy = true
        lock.unlock()

        let buffer = frame.pixelBuffer
        let time = frame.captureTime
        let k = frame.intrinsics, size = frame.size
        queue.async { [self] in
            defer {
                lock.lock()
                feedBusy = false
                lock.unlock()
            }
            if slam == nil { create(intrinsics: k, size: size) }
            guard let slam, downscaleLuma(buffer) else { return }
            scaled.withUnsafeBufferPointer {
                weee_slam_push_image(slam, time, $0.baseAddress, Int32(Self.width), Int32(Self.height), Int32(Self.width))
            }
            fedImages += 1
        }
    }

    // MARK: Tracker queue

    private func create(intrinsics k: SIMD4<Float>, size: SIMD2<Float>) {
        lock.lock()
        let s = settings
        lock.unlock()
        var config = WeeeSLAMConfig()
        config.width = Int32(Self.width)
        config.height = Int32(Self.height)
        // Kalibr calibration when present (scripts/calib), else AVFoundation's intrinsics and the
        // nominal extrinsics. The calibration-screen sliders trim on top of either.
        let c = UltraWideCalibration.self
        let calibrated = c.available && c.width == Self.width && c.height == Self.height
            && size == SIMD2(1440, 1080)
        if calibrated {
            config.fx = c.fx; config.fy = c.fy; config.cx = c.cx; config.cy = c.cy
            config.distortion = c.distortion
        } else {
            let sx = Double(Self.width) / Double(size.x), sy = Double(Self.height) / Double(size.y)
            config.fx = Double(k.x) * sx
            config.fy = Double(k.y) * sy
            config.cx = (Double(k.z) + 0.5) * sx - 0.5
            config.cy = (Double(k.w) + 0.5) * sy - 0.5
        }
        config.q_bc = calibrated ? c.qBC : (-0.7071068, 0.7071068, 0, 0)
        let p = calibrated ? c.pBC : (0, 0, 0)
        config.p_bc = (p.0 + Double(s.leverArmMM.x) / 1000, p.1 + Double(s.leverArmMM.y) / 1000,
                       p.2 + Double(s.leverArmMM.z) / 1000)
        config.time_offset = (calibrated ? c.timeOffset : 0) + Double(s.timeOffsetMS) / 1000
        Log.info("xrslam: using \(calibrated ? "Kalibr calibration" : "AVFoundation intrinsics, nominal extrinsics")")
        config.window_size = Int32(s.window)
        config.tracker_frequent = 3
        config.max_keypoints = Int32(s.keypoints)
        config.min_keypoint_distance = 25
        config.solver_iterations = 10
        config.solver_time_limit = 0.1
        let created = weee_slam_create(&config)
        lock.lock()
        slam = created
        lock.unlock()
        Log.info(String(format: "xrslam: %@ fx=%.1f fy=%.1f cx=%.1f cy=%.1f from %.0fx%.0f, offset=%.1f ms, lever=%.0f,%.0f,%.0f mm",
                        created == nil ? "create FAILED" : "created", config.fx, config.fy, config.cx, config.cy,
                        size.x, size.y, s.timeOffsetMS, s.leverArmMM.x, s.leverArmMM.y, s.leverArmMM.z))
    }

    private func destroy() {
        lock.lock()
        let old = slam
        slam = nil
        lock.unlock()
        if let old { weee_slam_destroy(old) }
        fedImages = 0
    }

    private func downscaleLuma(_ buffer: CVPixelBuffer) -> Bool {
        scaled.withUnsafeMutableBytes { Self.downscaleLuma(buffer, into: $0.baseAddress!) }
    }

    /// Y plane of the 4:2:0 camera buffer -> 640×480 gray at `destination`. The calibration
    /// recorder uses this too, so calibration sees exactly the tracker's images.
    static func downscaleLuma(_ buffer: CVPixelBuffer, into destination: UnsafeMutableRawPointer) -> Bool {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddressOfPlane(buffer, 0) else { return false }
        var source = vImage_Buffer(data: base,
                                   height: vImagePixelCount(CVPixelBufferGetHeightOfPlane(buffer, 0)),
                                   width: vImagePixelCount(CVPixelBufferGetWidthOfPlane(buffer, 0)),
                                   rowBytes: CVPixelBufferGetBytesPerRowOfPlane(buffer, 0))
        var target = vImage_Buffer(data: destination, height: vImagePixelCount(height),
                                   width: vImagePixelCount(width), rowBytes: width)
        return vImageScale_Planar8(&source, &target, nil, vImage_Flags(kvImageNoFlags)) == kvImageNoError
    }

    /// The rotation about world +y closest to `r` (both worlds are gravity-aligned, so they differ
    /// only in heading).
    private static func headingOnly(_ r: simd_double3x3) -> simd_double3x3 {
        Frames.rotationY(Frames.yaw(of: r))
    }
}
