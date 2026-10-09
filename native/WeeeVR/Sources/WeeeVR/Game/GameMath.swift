import Foundation

// Game maths in Double on the standard library's SIMD types only (no `simd` module), so the game
// core also builds and runs its self-test on Linux (scripts/game-test).

typealias V3 = SIMD3<Double>

@inline(__always) func vdot(_ a: V3, _ b: V3) -> Double { (a * b).sum() }
@inline(__always) func vcross(_ a: V3, _ b: V3) -> V3 {
    V3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)
}
@inline(__always) func vlength(_ a: V3) -> Double { vdot(a, a).squareRoot() }
@inline(__always) func vnormalize(_ a: V3) -> V3 {
    let l = vlength(a)
    return l > 1e-12 ? a / l : .zero
}
@inline(__always) func clamped(_ x: Double, _ lo: Double, _ hi: Double) -> Double { min(max(x, lo), hi) }

/// `v` rotated by the rotation vector `w` (axis * angle), Rodrigues' formula.
func rotate(_ v: V3, by w: V3) -> V3 {
    let angle = vlength(w)
    guard angle > 1e-12 else { return v }
    let k = w / angle, c = cos(angle), s = sin(angle)
    return v * c + vcross(k, v) * s + k * (vdot(k, v) * (1 - c))
}

/// Orthonormal right-handed frame: its axes in the parent frame (x × y = z).
struct Basis {
    var x: V3
    var y: V3
    var z: V3

    static let identity = Basis(x: V3(1, 0, 0), y: V3(0, 1, 0), z: V3(0, 0, 1))

    /// Local -> parent.
    func apply(_ p: V3) -> V3 { x * p.x + y * p.y + z * p.z }
    /// Parent -> local.
    func local(_ p: V3) -> V3 { V3(vdot(x, p), vdot(y, p), vdot(z, p)) }

    func rotated(by w: V3) -> Basis { Basis(x: rotate(x, by: w), y: rotate(y, by: w), z: rotate(z, by: w)) }

    /// The frame with `y` along `axis` and `z` as close to `normal` as possible.
    static func make(y axis: V3, z normal: V3) -> Basis {
        let y = vnormalize(axis)
        let z = vnormalize(normal - y * vdot(normal, y))
        return Basis(x: vcross(y, z), y: y, z: z)
    }

    /// Rotation vector taking this frame to `other` (other = rotate(self)).
    func rotationVector(to other: Basis) -> V3 {
        // R = other * self^T, column by column.
        let c0 = other.x * x.x + other.y * y.x + other.z * z.x
        let c1 = other.x * x.y + other.y * y.y + other.z * z.y
        let c2 = other.x * x.z + other.y * y.z + other.z * z.z
        let trace = c0.x + c1.y + c2.z
        let skew = V3(c1.z - c2.y, c2.x - c0.z, c0.y - c1.x)       // 2 sin(angle) * axis
        let angle = acos(clamped((trace - 1) / 2, -1, 1))
        if angle < 1e-6 { return skew / 2 }
        if angle > .pi - 1e-3 {
            // Near 180 degrees the skew part vanishes: axis from the symmetric part.
            let d = V3(c0.x, c1.y, c2.z)
            var axis = V3((max(d.x + 1, 0) / 2).squareRoot(), (max(d.y + 1, 0) / 2).squareRoot(),
                          (max(d.z + 1, 0) / 2).squareRoot())
            if c0.y + c1.x < 0 { axis.y = -axis.y }
            if c0.z + c2.x < 0 { axis.z = -axis.z }
            return vnormalize(axis) * angle
        }
        return skew / (2 * sin(angle)) * angle
    }
}

/// Minimum-jerk easing 0...1.
@inline(__always) func minimumJerk(_ s: Double) -> Double {
    let t = clamped(s, 0, 1)
    return t * t * t * (10 + t * (-15 + 6 * t))
}

/// Cubic Hermite between (p0, v0) at 0 and (p1, v1) at `duration`, evaluated at `t`.
func hermite(_ p0: V3, _ v0: V3, _ p1: V3, _ v1: V3, duration: Double, at t: Double) -> V3 {
    let s = clamped(t / max(duration, 1e-6), 0, 1)
    let s2 = s * s, s3 = s2 * s
    return p0 * (2 * s3 - 3 * s2 + 1) + v0 * ((s3 - 2 * s2 + s) * duration)
        + p1 * (-2 * s3 + 3 * s2) + v1 * ((s3 - s2) * duration)
}

/// Small deterministic random generator (the self-test is reproducible).
struct GameRandom {
    private var state: UInt64
    init(seed: UInt64) { state = seed &+ 0x9E37_79B9_7F4A_7C15 }
    mutating func next() -> Double {
        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        return Double(state >> 11) / Double(1 << 53)
    }
    mutating func uniform(_ lo: Double, _ hi: Double) -> Double { lo + (hi - lo) * next() }
}
