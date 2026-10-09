import Metal
import QuartzCore
import simd

/// Mirrors `Uniforms` in ShaderSource.swift — float4 fields only, so the layouts match exactly.
struct Uniforms {
    var screen = SIMD4<Float>()
    var lens = SIMD4<Float>()
    var optics = SIMD4<Float>()
    var camK = SIMD4<Float>()
    var cam = SIMD4<Float>()
    var camRot0 = SIMD4<Float>(1, 0, 0, 0)
    var camRot1 = SIMD4<Float>(0, 1, 0, 0)
    var camRot2 = SIMD4<Float>(0, 0, 1, 0)
    var camPos = SIMD4<Float>()
    var world0 = SIMD4<Float>(1, 0, 0, 0)
    var world1 = SIMD4<Float>(0, 1, 0, 0)
    var world2 = SIMD4<Float>(0, 0, 1, 0)
    var head = SIMD4<Float>(0, 0, 0, 1.5)
    var cube = SIMD4<Float>()
    var hud = SIMD4<Float>()
    var image = SIMD4<Float>()      // YCbCr matrix (0 BT.601, 1 BT.709), sharpening, black lift, edge softness
    var chroma = SIMD4<Float>()     // red / blue tan-angle scale - 1 (lens CA), P3 colour fix (0/1), 0

    mutating func setCameraFromWorld(_ m: simd_float3x3) {
        camRot0 = SIMD4(m.columns.0, 0); camRot1 = SIMD4(m.columns.1, 0); camRot2 = SIMD4(m.columns.2, 0)
    }

    mutating func setWorldFromView(_ m: simd_float3x3) {
        world0 = SIMD4(m.columns.0, 0); world1 = SIMD4(m.columns.1, 0); world2 = SIMD4(m.columns.2, 0)
    }
}

/// GPU side only: one pipeline, one full-screen draw per frame.
final class Renderer {
    let device: MTLDevice
    private let queue: MTLCommandQueue
    private let pipeline: MTLRenderPipelineState
    private let emptyTexture: MTLTexture

    private let hudLock = NSLock()
    private var hudTexture: MTLTexture?

    init(device: MTLDevice, pixelFormat: MTLPixelFormat) throws {
        self.device = device
        guard let queue = device.makeCommandQueue() else { throw CameraError("no Metal command queue") }
        self.queue = queue

        let library = try device.makeLibrary(source: ShaderSource.metal, options: nil)
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = library.makeFunction(name: "vrVertex")
        descriptor.fragmentFunction = library.makeFunction(name: "vrFragment")
        descriptor.colorAttachments[0].pixelFormat = pixelFormat
        pipeline = try device.makeRenderPipelineState(descriptor: descriptor)

        let empty = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: 1, height: 1, mipmapped: false)
        guard let texture = device.makeTexture(descriptor: empty) else { throw CameraError("no Metal texture") }
        emptyTexture = texture
    }

    /// The HUD text texture (premultiplied RGBA), swapped in from the main thread; nil hides the HUD.
    func setHUD(_ texture: MTLTexture?) {
        hudLock.lock()
        hudTexture = texture
        hudLock.unlock()
    }

    var hasHUD: Bool {
        hudLock.lock()
        defer { hudLock.unlock() }
        return hudTexture != nil
    }

    /// Encodes and presents one frame. `frame` is retained until the GPU has finished with it.
    /// `completed` reports the GPU execution time (seconds).
    func draw(to drawable: CAMetalDrawable, frame: CameraFrame?, uniforms: Uniforms, hands: [SIMD4<Float>],
              scene: [SIMD4<Float>],
              presented: @escaping (Double) -> Void, completed: @escaping (Double) -> Void) {
        guard let commandBuffer = queue.makeCommandBuffer() else { return }

        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = drawable.texture
        pass.colorAttachments[0].storeAction = .store

        if let frame, let luma = frame.lumaTexture, let chroma = frame.chromaTexture {
            hudLock.lock()
            let hud = hudTexture ?? emptyTexture
            hudLock.unlock()

            pass.colorAttachments[0].loadAction = .dontCare   // the triangle covers every pixel
            if let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) {
                var u = uniforms
                encoder.setRenderPipelineState(pipeline)
                encoder.setFragmentBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 0)
                hands.withUnsafeBytes { encoder.setFragmentBytes($0.baseAddress!, length: $0.count, index: 1) }
                scene.withUnsafeBytes { encoder.setFragmentBytes($0.baseAddress!, length: $0.count, index: 2) }
                encoder.setFragmentTexture(luma, index: 0)
                encoder.setFragmentTexture(chroma, index: 1)
                encoder.setFragmentTexture(hud, index: 2)
                encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
                encoder.endEncoding()
            }
            commandBuffer.addCompletedHandler { buffer in
                withExtendedLifetime(frame) {}
                withExtendedLifetime(hud) {}
                completed(max(buffer.gpuEndTime - buffer.gpuStartTime, 0))
            }
        } else {
            pass.colorAttachments[0].loadAction = .clear
            pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
            commandBuffer.makeRenderCommandEncoder(descriptor: pass)?.endEncoding()
        }

        drawable.addPresentedHandler { d in
            if d.presentedTime > 0 { presented(d.presentedTime) }
        }
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
}
