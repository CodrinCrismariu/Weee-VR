import simd

/// Coordinate frames used throughout:
/// - device: CoreMotion's portrait device frame (x right, y toward the top of the phone, z out of the screen).
/// - view:   the landscape-right screen frame (x right, y up, -z forward), identical to ARKit's camera
///           frame and to the back camera's native buffer orientation.
/// - world:  y up. ARKit's world frame, or CoreMotion's z-up reference frame turned y-up.
enum Frames {
    /// view -> device. In landscape-right the top of the phone points left,
    /// so screen-right = device -y and screen-up = device +x.
    static let viewToDevice = simd_double3x3(columns: (
        SIMD3(0, -1, 0),
        SIMD3(1, 0, 0),
        SIMD3(0, 0, 1)
    ))

    /// CoreMotion z-up reference frame -> y-up world frame: (x, y, z) -> (x, z, -y).
    static let zUpToYUp = simd_double3x3(columns: (
        SIMD3(1, 0, 0),
        SIMD3(0, 0, -1),
        SIMD3(0, 1, 0)
    ))

    static func rotationY(_ angle: Double) -> simd_double3x3 {
        let c = cos(angle), s = sin(angle)
        return simd_double3x3(columns: (SIMD3(c, 0, -s), SIMD3(0, 1, 0), SIMD3(s, 0, c)))
    }

    /// Heading of a rotation's forward (-z) axis around world +y.
    static func yaw(of r: simd_double3x3) -> Double {
        let f = r * SIMD3(0, 0, -1)
        return atan2(-f.x, -f.z)
    }
}

extension simd_double3x3 {
    var float: simd_float3x3 {
        simd_float3x3(columns: (
            SIMD3<Float>(columns.0), SIMD3<Float>(columns.1), SIMD3<Float>(columns.2)
        ))
    }
}

extension simd_float4x4 {
    var rotation: simd_double3x3 {
        simd_double3x3(columns: (
            SIMD3<Double>(SIMD3<Float>(columns.0.x, columns.0.y, columns.0.z)),
            SIMD3<Double>(SIMD3<Float>(columns.1.x, columns.1.y, columns.1.z)),
            SIMD3<Double>(SIMD3<Float>(columns.2.x, columns.2.y, columns.2.z))
        ))
    }

    var translation: SIMD3<Float> { SIMD3(columns.3.x, columns.3.y, columns.3.z) }
}
