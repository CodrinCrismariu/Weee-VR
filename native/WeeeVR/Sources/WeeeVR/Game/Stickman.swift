import Foundation

/// A stick figure playing the opponent's paddle, in the table frame (drawn by the shader's `stickHit`;
/// no physics). It stands behind the far end beside the paddle, on the side of the paddle's handle:
/// - its arm reaches the handle (two-bone IK, the elbow out and down), and it shuffles over to keep the
///   paddle within reach, crouching for low balls and leaning into the shot;
/// - its feet stay planted and step (one at a time, a small lift) when the body moves away from them;
/// - the free arm hangs out for balance and swings against the paddle.
struct Stickman {
    /// head (w: radius), neck, pelvis, paddle shoulder, elbow, hand, free shoulder, elbow, hand,
    /// knee, foot, knee, foot.
    static let jointCount = 13
    static let headRadius = 0.11
    private(set) var joints = [V3](repeating: .zero, count: jointCount)

    private var body: V3?                       // pelvis over the floor (x, z)
    private var feet = [Foot(), Foot()]
    private var time: Double?
    private var lastHand: V3?
    private var handSpeed = V3.zero

    private struct Foot {
        var at: V3?
        var from = V3.zero, to = V3.zero
        var start: Double?
    }

    static let upperArm = 0.31, forearm = 0.30
    static let thigh = 0.48, shin = 0.47
    static let stepTime = 0.17

    mutating func update(paddle: PaddleState, time t: Double) {
        let dt = time.map { min(max(t - $0, 0), 0.1) } ?? 0
        time = t
        let floor = Table.floorY
        let up = V3(0, 1, 0)
        let grip = paddle.center - paddle.basis.y * (PaddleShape.semiLength + PaddleShape.handleLength * 0.6)
        let side: Double = grip.x >= paddle.center.x ? 1 : -1       // the body is on the handle's side

        // The body: beside the paddle, behind the end line; it follows smoothly.
        let want = V3(paddle.center.x + side * 0.40, 0, min(paddle.center.z - 0.30, -Table.halfLength - 0.35))
        var b = body ?? want
        b += (want - b) * (dt > 0 ? 1 - exp(-dt / 0.22) : 1)

        // Crouch for a low paddle; then step over so the shoulder can reach the grip.
        let standingShoulder = floor + 1.44
        let crouch = clamped(standingShoulder - grip.y - 0.40, 0, 0.28)
        let shoulderY = standingShoulder - crouch
        let reach = (Self.upperArm + Self.forearm) * 0.97
        var shoulder = V3(b.x - side * 0.15, shoulderY, b.z)
        let toGrip = grip - shoulder
        let flat = V3(toGrip.x, 0, toGrip.z)
        let maxFlat = sqrt(max(reach * reach - toGrip.y * toGrip.y, 0.01))
        if vlength(flat) > maxFlat { b += vnormalize(flat) * (vlength(flat) - maxFlat) }
        body = b

        // Torso, leaning a little into the shot.
        let lean = V3(flat.x, 0, flat.z) * 0.18
        let pelvis = V3(b.x, floor + 0.93 - crouch, b.z)
        let neck = pelvis + up * 0.55 + lean
        let head = neck + up * 0.15 + lean * 0.3
        shoulder = neck + V3(-side * 0.15, -0.05, 0)
        let freeShoulder = neck + V3(side * 0.15, -0.05, 0)

        // The paddle arm: the hand on the grip, the elbow out and down.
        let arm = Self.ik(shoulder, grip, Self.upperArm, Self.forearm, pole: V3(-side * 0.6, -1, -0.35))
        // The free arm out for balance, swinging against the paddle hand.
        if let last = lastHand, dt > 0 { handSpeed += ((arm.end - last) / dt - handSpeed) * min(dt / 0.08, 1) }
        lastHand = arm.end
        let counter = V3(handSpeed.x, 0, handSpeed.z) * -0.04
        let freeTarget = freeShoulder + V3(side * 0.16, -0.42, 0.16) + V3(clamped(counter.x, -0.15, 0.15), 0, clamped(counter.z, -0.15, 0.15))
        let free = Self.ik(freeShoulder, freeTarget, Self.upperArm, Self.forearm, pole: V3(side * 0.5, -1, -0.6))

        // Feet: planted; one steps when the body has moved off it.
        for i in 0..<2 {
            let rest = V3(b.x + (i == 0 ? -0.16 : 0.16), floor + 0.02, b.z + 0.03)
            var f = feet[i]
            if f.at == nil { f.at = rest }
            let otherStepping = feet[1 - i].start != nil
            if f.start == nil, !otherStepping, vlength(rest - f.at!) > 0.18 {
                f.from = f.at!
                f.to = rest + (rest - f.at!) * 0.25                     // a little past, as the body keeps going
                f.start = t
            }
            if let start = f.start {
                let k = (t - start) / Self.stepTime
                if k >= 1 {
                    f.at = f.to
                    f.start = nil
                } else {
                    let s = k * k * (3 - 2 * k)
                    var p = f.from + (f.to - f.from) * s
                    p.y = floor + 0.02 + 0.07 * sin(.pi * k)
                    f.at = p
                }
            }
            feet[i] = f
        }
        let legR = Self.ik(pelvis, feet[0].at!, Self.thigh, Self.shin, pole: V3(0, 0.3, 1))
        let legL = Self.ik(pelvis, feet[1].at!, Self.thigh, Self.shin, pole: V3(0, 0.3, 1))

        joints = [head, neck, pelvis, shoulder, arm.elbow, arm.end, freeShoulder, free.elbow, free.end,
                  legR.elbow, legR.end, legL.elbow, legL.end]
    }

    /// Two bones from `root` toward `target` (lengths `a`, `b`): the middle joint bends toward `pole`.
    /// Out of reach, the chain points at the target, straight.
    static func ik(_ root: V3, _ target: V3, _ a: Double, _ b: Double, pole: V3) -> (elbow: V3, end: V3) {
        let d = target - root
        let length = vlength(d)
        let direction = length > 1e-6 ? d / length : V3(0, -1, 0)
        let dist = clamped(length, abs(a - b) + 1e-3, a + b - 1e-4)
        let cosA = clamped((a * a + dist * dist - b * b) / (2 * a * dist), -1, 1)
        let sinA = sqrt(max(1 - cosA * cosA, 0))
        var bend = pole - direction * vdot(pole, direction)
        if vlength(bend) < 1e-6 { bend = vcross(direction, V3(1, 0, 0)) }
        bend = vnormalize(bend)
        return (root + direction * (a * cosA) + bend * (a * sinA), root + direction * dist)
    }
}
