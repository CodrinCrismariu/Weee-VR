import CoreVideo
import Foundation
import Metal
import QuartzCore
import simd

/// One camera image, already wrapped as Metal textures (zero-copy, via CVMetalTextureCache).
struct CameraFrame {
    let pixelBuffer: CVPixelBuffer      // keeps the IOSurface alive while the GPU reads it
    let luma: CVMetalTexture
    let chroma: CVMetalTexture
    let size: SIMD2<Float>              // buffer pixels
    let intrinsics: SIMD4<Float>        // fx, fy, cx, cy in buffer pixels
    let captureTime: Double             // host clock (same base as CACurrentMediaTime)
    let arrivalTime: Double             // when the app received it
    var pose: simd_float4x4?            // world <- camera at capture time from ARKit/XRSLAM (nil = gyro 3DoF)
    var sequence: UInt64 = 0            // assigned by FrameMailbox.put
    var velocity = SIMD3<Float>.zero    // camera velocity (m/s, world), smoothed
    var trackingNormal = false          // the 6DoF tracker has a good position
    var proxyDepth: Float?              // auto depth-proxy distance from the tracker's feature points (m)

    var lumaTexture: MTLTexture? { CVMetalTextureGetTexture(luma) }
    var chromaTexture: MTLTexture? { CVMetalTextureGetTexture(chroma) }
}

/// Latest-frame-wins handoff from the camera thread to the render thread.
final class FrameMailbox {
    private let lock = NSLock()
    private var latest: CameraFrame?
    private var nextSequence: UInt64 = 0
    private var lastRendered: UInt64 = .max
    private(set) var received: UInt64 = 0
    private(set) var neverShown: UInt64 = 0

    func put(_ frame: CameraFrame) {
        lock.lock()
        defer { lock.unlock() }
        if let latest, latest.sequence != lastRendered { neverShown += 1 }
        var frame = frame
        frame.sequence = nextSequence
        latest = frame
        nextSequence += 1
        received += 1
    }

    /// The newest frame (may be the same one as last time if the camera hasn't delivered yet).
    func take() -> CameraFrame? {
        lock.lock()
        defer { lock.unlock() }
        if let latest { lastRendered = latest.sequence }
        return latest
    }

    func clear() {
        lock.lock()
        latest = nil
        lock.unlock()
    }

    func counters() -> (received: UInt64, neverShown: UInt64) {
        lock.lock()
        defer { lock.unlock() }
        return (received, neverShown)
    }
}

final class TextureMaker {
    private var cache: CVMetalTextureCache?

    init(device: MTLDevice) {
        CVMetalTextureCacheCreate(nil, nil, device, nil, &cache)
    }

    /// Wraps a bi-planar 4:2:0 YCbCr buffer as (Y: r8, CbCr: rg8) textures.
    func planes(of buffer: CVPixelBuffer) -> (CVMetalTexture, CVMetalTexture)? {
        guard let cache, CVPixelBufferGetPlaneCount(buffer) >= 2 else { return nil }
        func plane(_ index: Int, _ format: MTLPixelFormat) -> CVMetalTexture? {
            var texture: CVMetalTexture?
            CVMetalTextureCacheCreateTextureFromImage(
                nil, cache, buffer, nil, format,
                CVPixelBufferGetWidthOfPlane(buffer, index),
                CVPixelBufferGetHeightOfPlane(buffer, index),
                index, &texture
            )
            return texture
        }
        guard let y = plane(0, .r8Unorm), let cbcr = plane(1, .rg8Unorm) else { return nil }
        return (y, cbcr)
    }
}
