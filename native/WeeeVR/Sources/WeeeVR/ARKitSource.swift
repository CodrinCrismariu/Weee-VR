import ARKit
import Foundation
import QuartzCore
import simd

/// 6DoF mode: ARKit world tracking, i.e. Apple's on-device visual-inertial SLAM (map + relocalisation).
/// Each frame carries the camera image, exact intrinsics, the camera pose at capture time, a smoothed
/// velocity for display-time prediction, and an auto depth-proxy distance from ARKit's feature points.
/// Everything here runs on ARKit's delegate queue, never on the render thread.
final class ARKitSource: NSObject, ARSessionDelegate {
    private let session = ARSession()
    private let queue = DispatchQueue(label: "arkit.frames", qos: .userInteractive)
    private let textures: TextureMaker
    private let mailbox: FrameMailbox

    private let lock = NSLock()
    private var _trackingState = "—"
    private var config: ARWorldTrackingConfiguration?

    // Delegate-queue state
    private var lastPosition: SIMD3<Float>?
    private var lastTime: Double = 0
    private var velocity = SIMD3<Float>.zero
    private var depth: Float?

    var onStatus: ((String) -> Void)?
    /// Every frame, on the ARKit delegate queue, after its pose is attached (hand tracking).
    var onFrame: ((CameraFrame) -> Void)?

    init(textures: TextureMaker, mailbox: FrameMailbox) {
        self.textures = textures
        self.mailbox = mailbox
        super.init()
        session.delegate = self
        session.delegateQueue = queue
    }

    var trackingState: String {
        lock.lock()
        defer { lock.unlock() }
        return _trackingState
    }

    func start() {
        guard ARWorldTrackingConfiguration.isSupported else {
            onStatus?("ARKit world tracking not supported")
            return
        }
        let config = Self.makeConfiguration()
        lock.lock()
        self.config = config
        lock.unlock()
        run(config)
    }

    /// Discards the map and the world origin: tracking restarts from the current pose.
    func reset() {
        lock.lock()
        let config = self.config
        lock.unlock()
        guard let config else { return }
        Log.info("reset: ARKit map and position")
        run(config)
    }

    func stop() {
        session.pause()
    }

    private func run(_ config: ARWorldTrackingConfiguration) {
        queue.async { [self] in
            lastPosition = nil
            velocity = .zero
            depth = nil
        }
        session.run(config, options: [.resetTracking, .removeExistingAnchors])
        let f = config.videoFormat
        Log.info(String(format: "arkit: %@ %ldx%ld@%ld", f.captureDeviceType.rawValue,
                        Int(f.imageResolution.width), Int(f.imageResolution.height), f.framesPerSecond))
        onStatus?("ARKit \(f.captureDeviceType == .builtInUltraWideCamera ? "UltraWide" : "Main") "
                  + "\(Int(f.imageResolution.height))p\(f.framesPerSecond)")
    }

    /// Tracking only: no planes, meshes, light or environment estimation, people or audio.
    /// Prefers the ultra-wide if ARKit offers it for world tracking, then 60 fps, then resolution.
    private static func makeConfiguration() -> ARWorldTrackingConfiguration {
        let config = ARWorldTrackingConfiguration()
        let formats = ARWorldTrackingConfiguration.supportedVideoFormats
        if let best = formats.max(by: { rank($0) < rank($1) }) { config.videoFormat = best }
        config.planeDetection = []
        config.environmentTexturing = .none
        config.isLightEstimationEnabled = false
        config.frameSemantics = []
        config.providesAudioData = false
        config.isCollaborationEnabled = false
        config.isAutoFocusEnabled = true
        return config
    }

    private static func rank(_ f: ARConfiguration.VideoFormat) -> Double {
        (f.captureDeviceType == .builtInUltraWideCamera ? 1e9 : 0)
            + (f.framesPerSecond >= 60 ? 1e8 : 0)
            + Double(f.imageResolution.width * f.imageResolution.height)
    }

    // MARK: ARSessionDelegate

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let arrival = CACurrentMediaTime()
        let buffer = frame.capturedImage
        guard let planes = textures.planes(of: buffer) else { return }
        let (luma, chroma) = planes

        let k = frame.camera.intrinsics
        let resolution = frame.camera.imageResolution
        let pose = frame.camera.transform
        let timestamp = frame.timestamp

        let normal: Bool
        let state: String
        switch frame.camera.trackingState {
        case .normal: normal = true; state = "tracking"
        case .notAvailable: normal = false; state = "no tracking"
        case .limited(let reason): normal = false; state = Self.describe(reason)
        }
        lock.lock()
        _trackingState = state
        lock.unlock()

        updateVelocity(position: pose.translation, time: timestamp, normal: normal)
        if normal, let measured = Self.medianDepth(points: frame.rawFeaturePoints?.points, cameraPose: pose) {
            let dt = Float(max(timestamp - lastTime, 0))
            let alpha = depth == nil ? 1 : 1 - exp(-dt / 0.3)
            depth = (depth ?? measured) + alpha * (measured - (depth ?? measured))
        }
        lastTime = timestamp

        let velocity = self.velocity, depth = self.depth
        var f = CameraFrame(pixelBuffer: buffer, luma: luma, chroma: chroma,
                            size: SIMD2(Float(resolution.width), Float(resolution.height)),
                            intrinsics: SIMD4(k.columns.0.x, k.columns.1.y, k.columns.2.x, k.columns.2.y),
                            captureTime: timestamp, arrivalTime: arrival, pose: pose)
        f.velocity = velocity
        f.trackingNormal = normal
        f.proxyDepth = depth
        f.ycbcrMatrix = CameraFrame.ycbcrMatrix(of: buffer)
        f.needsP3Conversion = CameraFrame.needsP3Conversion(buffer)
        mailbox.put(f)
        onFrame?(f)
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        onStatus?("ARKit error: \(error.localizedDescription)")
    }

    // MARK: Helpers (delegate queue)

    /// Exponentially smoothed velocity (time constant 0.1 s); zero while tracking is not normal.
    private func updateVelocity(position: SIMD3<Float>, time: Double, normal: Bool) {
        defer { lastPosition = normal ? position : nil }
        guard normal, let last = lastPosition else {
            velocity = .zero
            return
        }
        let dt = Float(time - lastTime)
        guard dt > 0.001, dt < 0.2 else { return }
        let instant = (position - last) / dt
        let alpha = 1 - exp(-dt / 0.1)
        velocity += alpha * (instant - velocity)
    }

    /// Median distance of ARKit feature points within ±20° of the camera axis, clamped to 0.4–6 m.
    private static func medianDepth(points: [SIMD3<Float>]?, cameraPose: simd_float4x4) -> Float? {
        guard let points, points.count >= 8 else { return nil }
        let cameraFromWorld = cameraPose.inverse
        let tan20: Float = 0.364
        var depths: [Float] = []
        depths.reserveCapacity(points.count)
        for p in points {
            let c = cameraFromWorld * SIMD4(p, 1)
            let z = -c.z
            if z > 0.2, abs(c.x) < tan20 * z, abs(c.y) < tan20 * z { depths.append(z) }
        }
        guard depths.count >= 5 else { return nil }
        depths.sort()
        return min(max(depths[depths.count / 2], 0.4), 6)
    }

    private static func describe(_ reason: ARCamera.TrackingState.Reason) -> String {
        switch reason {
        case .initializing: "mapping…"
        case .relocalizing: "relocalizing…"
        case .excessiveMotion: "limited: moving too fast"
        case .insufficientFeatures: "limited: not enough detail"
        @unknown default: "limited"
        }
    }
}
