import Metal
import UIKit

/// Draws the stats text into a small premultiplied-RGBA texture that the passthrough shader composites
/// above each lens in the same pass. A UIKit/SwiftUI layer on top of the CAMetalLayer would make the
/// compositor blend layers and can cost a display refresh. ~52 characters per line, 7 lines.
enum HUDRasterizer {
    static let size = SIMD2<Float>(720, 240)   // drawable pixels

    static func makeTexture(text: String, device: MTLDevice) -> MTLTexture? {
        let width = Int(size.x), height = Int(size.y)
        let bytesPerRow = width * 4
        var pixels = [UInt8](repeating: 0, count: bytesPerRow * height)

        let drawn: Bool = pixels.withUnsafeMutableBytes { buffer in
            guard let context = CGContext(
                data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 8,
                bytesPerRow: bytesPerRow, space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            ) else { return false }
            context.translateBy(x: 0, y: CGFloat(height))   // UIKit's top-left origin
            context.scaleBy(x: 1, y: -1)
            UIGraphicsPushContext(context)
            UIColor(white: 0, alpha: 0.6).setFill()
            UIBezierPath(roundedRect: CGRect(x: 0, y: 0, width: width, height: height), cornerRadius: 12).fill()
            (text as NSString).draw(
                in: CGRect(x: 12, y: 10, width: width - 24, height: height - 20),
                withAttributes: [
                    .font: UIFont.monospacedSystemFont(ofSize: 21, weight: .medium),
                    .foregroundColor: UIColor.white,
                ]
            )
            UIGraphicsPopContext()
            return true
        }
        guard drawn else { return nil }

        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm, width: width, height: height, mipmapped: false)
        descriptor.usage = .shaderRead
        guard let texture = device.makeTexture(descriptor: descriptor) else { return nil }
        texture.replace(region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0,
                        withBytes: pixels, bytesPerRow: bytesPerRow)
        return texture
    }
}
