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

/// The scoreboard beside the net (ShaderSource `kScore*`): the score, you on the left, and a line under
/// it (game point, the winner, recording). Premultiplied RGBA, 2:1 like the board.
enum ScoreRasterizer {
    static func makeTexture(player: Int, opponent: Int, message: String, recording: Bool, device: MTLDevice) -> MTLTexture? {
        let width = 512, height = 256
        let bytesPerRow = width * 4
        var pixels = [UInt8](repeating: 0, count: bytesPerRow * height)
        let drawn: Bool = pixels.withUnsafeMutableBytes { buffer in
            guard let context = CGContext(
                data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 8,
                bytesPerRow: bytesPerRow, space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            ) else { return false }
            context.translateBy(x: 0, y: CGFloat(height))
            context.scaleBy(x: 1, y: -1)
            UIGraphicsPushContext(context)
            UIColor(white: 0.05, alpha: 0.78).setFill()
            UIBezierPath(roundedRect: CGRect(x: 2, y: 2, width: width - 4, height: height - 4), cornerRadius: 26).fill()
            let centred = NSMutableParagraphStyle()
            centred.alignment = .center
            func text(_ s: String, _ rect: CGRect, size: CGFloat, weight: UIFont.Weight, colour: UIColor, mono: Bool = false) {
                let font = mono ? UIFont.monospacedDigitSystemFont(ofSize: size, weight: weight) : UIFont.systemFont(ofSize: size, weight: weight)
                (s as NSString).draw(in: rect, withAttributes: [.font: font, .foregroundColor: colour, .paragraphStyle: centred])
            }
            let half = CGFloat(width) / 2
            text("YOU", CGRect(x: 0, y: 14, width: half, height: 40), size: 30, weight: .semibold, colour: UIColor(white: 0.75, alpha: 1))
            text("AI", CGRect(x: half, y: 14, width: half, height: 40), size: 30, weight: .semibold, colour: UIColor(white: 0.75, alpha: 1))
            text("\(player)", CGRect(x: 0, y: 44, width: half, height: 140), size: 118, weight: .bold, colour: .white, mono: true)
            text("\(opponent)", CGRect(x: half, y: 44, width: half, height: 140), size: 118, weight: .bold, colour: .white, mono: true)
            text(":", CGRect(x: half - 30, y: 56, width: 60, height: 120), size: 90, weight: .bold, colour: UIColor(white: 0.6, alpha: 1))
            text(message, CGRect(x: 0, y: 190, width: CGFloat(width), height: 44), size: 30, weight: .heavy,
                 colour: UIColor(red: 1, green: 0.82, blue: 0.25, alpha: 1))
            if recording {
                UIColor(red: 1, green: 0.15, blue: 0.15, alpha: 1).setFill()
                UIBezierPath(ovalIn: CGRect(x: 22, y: 22, width: 18, height: 18)).fill()
            }
            UIGraphicsPopContext()
            return true
        }
        guard drawn else { return nil }
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: width, height: height, mipmapped: false)
        descriptor.usage = .shaderRead
        guard let texture = device.makeTexture(descriptor: descriptor) else { return nil }
        texture.replace(region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0, withBytes: pixels, bytesPerRow: bytesPerRow)
        return texture
    }
}
