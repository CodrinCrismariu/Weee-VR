import AVFoundation
import CoreMedia
import Foundation
import QuartzCore
import simd

/// Lowest-latency passthrough: raw AVFoundation capture with stabilisation off, fixed frame rate,
/// late frames discarded, and 4:2:0 full-range buffers handed straight to Metal.
final class AVCameraSource: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    enum Lens: String, CaseIterable {
        case main = "Main"
        case ultraWide = "UltraWide"

        var deviceType: AVCaptureDevice.DeviceType {
            switch self {
            case .main: .builtInWideAngleCamera
            case .ultraWide: .builtInUltraWideCamera
            }
        }
    }

    struct Preset: Equatable {
        let width: Int32
        let height: Int32
        let fps: Int32
        var name: String { "\(width)x\(height)@\(fps)" }
    }

    /// 4:3 first: a Cardboard eye view is roughly square, and 16:9 crops ~16° off the ultra-wide's height.
    static let presets = [
        Preset(width: 1440, height: 1080, fps: 60),
        Preset(width: 1920, height: 1080, fps: 60),
        Preset(width: 1280, height: 720, fps: 60),
        Preset(width: 1920, height: 1080, fps: 120),
        Preset(width: 1280, height: 720, fps: 120),
    ]

    private let session = AVCaptureSession()
    private let output = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "camera.session")
    private let frameQueue = DispatchQueue(label: "camera.frames", qos: .userInteractive)
    private let textures: TextureMaker
    private let mailbox: FrameMailbox

    private let lock = NSLock()
    private var fallbackIntrinsics = SIMD4<Float>(1, 1, 0, 0)  // from the format's FOV
    private var horizontalFov: Float = 0
    private var _usingAttachedIntrinsics = false

    var onStatus: ((String) -> Void)?
    private var _frameHook: ((inout CameraFrame) -> Void)?

    /// Called on the capture queue for every frame before it reaches the renderer (XRSLAM attaches
    /// its pose here and takes a copy of the image for tracking).
    var frameHook: ((inout CameraFrame) -> Void)? {
        get { lock.lock(); defer { lock.unlock() }; return _frameHook }
        set { lock.lock(); _frameHook = newValue; lock.unlock() }
    }

    init(textures: TextureMaker, mailbox: FrameMailbox) {
        self.textures = textures
        self.mailbox = mailbox
        super.init()
    }

    var usingAttachedIntrinsics: Bool {
        lock.lock()
        defer { lock.unlock() }
        return _usingAttachedIntrinsics
    }

    func start(lens: Lens, preset: Preset) {
        sessionQueue.async { [self] in
            do {
                let status = try configure(lens: lens, preset: preset)
                if !session.isRunning { session.startRunning() }
                onStatus?(status)
            } catch {
                onStatus?("camera error: \(error.localizedDescription)")
            }
        }
    }

    func stop() {
        sessionQueue.async { [self] in
            if session.isRunning { session.stopRunning() }
        }
    }

    /// Calibration: cap exposure at `maxSeconds` (ISO raised to keep brightness) so the target stays
    /// sharp while the phone moves; `nil` returns to auto exposure.
    func limitExposure(to maxSeconds: Double?) {
        sessionQueue.async { [self] in
            guard let device = (session.inputs.first as? AVCaptureDeviceInput)?.device,
                  (try? device.lockForConfiguration()) != nil else { return }
            defer { device.unlockForConfiguration() }
            guard let maxSeconds else {
                if device.isExposureModeSupported(.continuousAutoExposure) { device.exposureMode = .continuousAutoExposure }
                return
            }
            guard device.isExposureModeSupported(.custom) else { return }
            let current = device.exposureDuration.seconds
            let target = min(current, maxSeconds)
            let format = device.activeFormat
            let iso = min(max(device.iso * Float(current / target), format.minISO), format.maxISO)
            let duration = CMTime(seconds: max(target, format.minExposureDuration.seconds), preferredTimescale: 1_000_000)
            device.setExposureModeCustom(duration: duration, iso: iso)
            Log.info(String(format: "camera: exposure %.2f ms -> %.2f ms, iso %.0f -> %.0f", current * 1000,
                            duration.seconds * 1000, device.iso, iso))
        }
    }

    private func configure(lens: Lens, preset: Preset) throws -> String {
        guard let device = AVCaptureDevice.default(lens.deviceType, for: .video, position: .back) else {
            throw CameraError("\(lens.rawValue) camera not available")
        }

        session.beginConfiguration()
        defer { session.commitConfiguration() }

        for input in session.inputs { session.removeInput(input) }
        let input = try AVCaptureDeviceInput(device: device)
        guard session.canAddInput(input) else { throw CameraError("cannot add camera input") }
        session.addInput(input)

        if session.outputs.isEmpty {
            output.videoSettings = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
            ]
            output.alwaysDiscardsLateVideoFrames = true
            output.setSampleBufferDelegate(self, queue: frameQueue)
            guard session.canAddOutput(output) else { throw CameraError("cannot add video output") }
            session.addOutput(output)
        }
        session.sessionPreset = .inputPriority

        guard let format = Self.bestFormat(for: device, preset: preset) else {
            throw CameraError("\(lens.rawValue) has no \(preset.name) format")
        }

        try device.lockForConfiguration()
        device.activeFormat = format
        let frameDuration = CMTime(value: 1, timescale: preset.fps)
        device.activeVideoMinFrameDuration = frameDuration   // min == max: fixed rate, AE can't
        device.activeVideoMaxFrameDuration = frameDuration   // stretch exposure in dim light
        if device.isLowLightBoostSupported { device.automaticallyEnablesLowLightBoostWhenAvailable = false }
        if format.isVideoHDRSupported {
            device.automaticallyAdjustsVideoHDREnabled = false
            device.isVideoHDREnabled = false
        }
        if device.isFocusModeSupported(.continuousAutoFocus) { device.focusMode = .continuousAutoFocus }
        // The shader assumes a pinhole camera; the ISP rectifies the ultra-wide's barrel distortion.
        let corrected = device.isGeometricDistortionCorrectionSupported
        if corrected { device.isGeometricDistortionCorrectionEnabled = true }
        device.unlockForConfiguration()

        if let connection = output.connection(with: .video) {
            if connection.isVideoStabilizationSupported {
                connection.preferredVideoStabilizationMode = .off   // stabilisation buffers frames
            }
            if connection.isCameraIntrinsicMatrixDeliverySupported {
                connection.isCameraIntrinsicMatrixDeliveryEnabled = true
            }
        }

        let dims = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
        let hfov = corrected ? format.geometricDistortionCorrectedVideoFieldOfView : format.videoFieldOfView
        let fx = Float(dims.width) / 2 / tan(hfov * .pi / 360)
        lock.lock()
        fallbackIntrinsics = SIMD4(fx, fx, Float(dims.width) / 2, Float(dims.height) / 2)
        horizontalFov = hfov
        lock.unlock()

        Log.info(String(format: "camera: %@ %dx%d@%d hfov=%.1f (uncorrected %.1f) gdc=%@ binned=%@", lens.rawValue,
              dims.width, dims.height, preset.fps, hfov, format.videoFieldOfView, corrected ? "yes" : "no",
              format.isVideoBinned ? "yes" : "no"))
        return "\(lens.rawValue) \(preset.name) hfov \(String(format: "%.0f", hfov))°"
    }

    /// 8-bit full-range 4:2:0 format at the preset size that can run at the preset fps.
    /// Prefers the widest field of view, then the lowest max frame rate (avoids cropped slo-mo modes).
    static func bestFormat(for device: AVCaptureDevice, preset: Preset) -> AVCaptureDevice.Format? {
        device.formats
            .filter { format in
                let desc = format.formatDescription
                let dims = CMVideoFormatDescriptionGetDimensions(desc)
                return CMFormatDescriptionGetMediaSubType(desc) == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
                    && dims.width == preset.width && dims.height == preset.height
                    && format.videoSupportedFrameRateRanges.contains { $0.maxFrameRate >= Double(preset.fps) }
            }
            .max { a, b in
                if a.videoFieldOfView != b.videoFieldOfView { return a.videoFieldOfView < b.videoFieldOfView }
                return maxFps(a) > maxFps(b)
            }
    }

    static func isAvailable(lens: Lens, preset: Preset) -> Bool {
        guard let device = AVCaptureDevice.default(lens.deviceType, for: .video, position: .back) else { return false }
        return bestFormat(for: device, preset: preset) != nil
    }

    /// Position of `lens` relative to the 1× camera in the landscape view frame (x right, y up, z toward
    /// the viewer), metres, from Apple's factory extrinsics. The eye offset is calibrated against the 1×.
    static func offsetFromMain(lens: Lens) -> SIMD3<Float> {
        guard lens != .main,
              let from = AVCaptureDevice.default(lens.deviceType, for: .video, position: .back),
              let main = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let data = AVCaptureDevice.extrinsicMatrix(from: from, to: main),
              data.count >= MemoryLayout<simd_float4x3>.size
        else { return .zero }
        // Column 3 is `lens`'s origin in the 1× camera frame (x right, y down, z forward), in millimetres.
        let t = data.withUnsafeBytes { $0.loadUnaligned(as: simd_float4x3.self) }.columns.3
        let offset = SIMD3(t.x, -t.y, -t.z) / 1000
        Log.info(String(format: "extrinsics: %@ relative to 1x = %.1f, %.1f, %.1f mm (view frame)", lens.rawValue,
                        offset.x * 1000, offset.y * 1000, offset.z * 1000))
        return simd_length(offset) < 0.04 ? offset : .zero
    }

    private static func maxFps(_ format: AVCaptureDevice.Format) -> Double {
        format.videoSupportedFrameRateRanges.map(\.maxFrameRate).max() ?? 0
    }

    // MARK: AVCaptureVideoDataOutputSampleBufferDelegate

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        let arrival = CACurrentMediaTime()
        guard let buffer = CMSampleBufferGetImageBuffer(sampleBuffer),
              let planes = textures.planes(of: buffer)
        else { return }
        let (luma, chroma) = planes

        let width = Float(CVPixelBufferGetWidth(buffer))
        let height = Float(CVPixelBufferGetHeight(buffer))
        let captureTime = CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds

        lock.lock()
        var intrinsics = fallbackIntrinsics
        lock.unlock()
        var attached = false
        if let data = CMGetAttachment(sampleBuffer, key: kCMSampleBufferAttachmentKey_CameraIntrinsicMatrix,
                                      attachmentModeOut: nil) as? Data,
           data.count >= MemoryLayout<simd_float3x3>.size {
            let k = data.withUnsafeBytes { $0.loadUnaligned(as: simd_float3x3.self) }
            // Only trust it if the principal point is near the buffer centre, i.e. it is
            // expressed in this buffer's pixel units (not the full sensor's).
            if abs(k.columns.2.x / width - 0.5) < 0.1, abs(k.columns.2.y / height - 0.5) < 0.1 {
                intrinsics = SIMD4(k.columns.0.x, k.columns.1.y, k.columns.2.x, k.columns.2.y)
                attached = true
            }
        }
        lock.lock()
        _usingAttachedIntrinsics = attached
        let hook = _frameHook
        lock.unlock()

        var frame = CameraFrame(pixelBuffer: buffer, luma: luma, chroma: chroma, size: SIMD2(width, height),
                                intrinsics: intrinsics, captureTime: captureTime, arrivalTime: arrival, pose: nil)
        hook?(&frame)
        mailbox.put(frame)
    }
}

struct CameraError: LocalizedError {
    let message: String
    init(_ message: String) { self.message = message }
    var errorDescription: String? { message }
}
