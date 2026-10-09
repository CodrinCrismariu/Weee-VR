import Foundation

/// Paddle geometry (the shader mirrors it). Paddle frame: origin at the blade centre, y along the
/// blade from the handle to the tip, z the forehand (red) face's normal, x = y × z.
enum PaddleShape {
    static let semiWidth = 0.075            // blade 150 x 158 mm
    static let semiLength = 0.079
    static let halfThickness = 0.006        // blade + rubbers
    static let handleLength = 0.10
    static let handleHalfWidth = 0.0125
    static let handleHalfThickness = 0.011
    /// Blade centre from the middle of the handle (where the palm holds it).
    static var gripReach: Double { semiLength + handleLength / 2 }
}

struct PaddleState {
    var center: V3
    var basis: Basis
    var velocity = V3.zero                   // of the centre, m/s
    var angular = V3.zero                    // rad/s
    var visible = false

    init(center: V3 = .zero, basis: Basis = .identity) {
        self.center = center
        self.basis = basis
    }

    /// The pose a `fraction` of the way to `next`, moving with the velocity between them.
    func interpolated(to next: PaddleState, fraction: Double, over duration: Double) -> PaddleState {
        let turn = basis.rotationVector(to: next.basis)
        var p = PaddleState(center: center + (next.center - center) * fraction, basis: basis.rotated(by: turn * fraction))
        p.velocity = (next.center - center) / max(duration, 1e-4)
        p.angular = turn / max(duration, 1e-4)
        p.visible = next.visible
        return p
    }

    /// Shakehand grip from a right hand's joints (MediaPipe order, table frame): the handle lies in
    /// the palm, the forehand face looks where the palm does, and the blade points from the palm
    /// toward the thumb side of the knuckles by `gripAngle`; `offset` moves it along the palm normal
    /// (negative: toward the back of the hand, inside the fist).
    static func grip(_ j: [V3], gripAngle: Double, offset: Double) -> PaddleState? {
        guard j.count >= 21 else { return nil }
        let wrist = j[0], knuckles = (j[5] + j[9] + j[13] + j[17]) / 4
        let palm = (j[0] + j[5] + j[9] + j[13] + j[17]) / 5
        let forward = vnormalize(knuckles - wrist)
        var across = j[5] - j[17]                            // toward the index (thumb) side
        across = vnormalize(across - forward * vdot(across, forward))
        guard vlength(forward) > 0.5, vlength(across) > 0.5 else { return nil }
        let palmNormal = vcross(across, forward)              // out of a right hand's palm
        let axis = forward * cos(gripAngle) + across * sin(gripAngle)
        let basis = Basis.make(y: axis, z: palmNormal)
        var state = PaddleState(center: palm + basis.y * PaddleShape.gripReach + basis.z * offset, basis: basis)
        state.visible = true
        return state
    }
}
