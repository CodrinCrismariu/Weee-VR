import Foundation

/// A 40 mm, 2.7 g table tennis ball (thin-shelled: I = 2/3 m r²), in the table frame.
struct Ball {
    static let radius = 0.02
    static let mass = 0.0027
    static let inertia = 2.0 / 3.0          // I / (m r²)

    var p: V3
    var v = V3.zero
    var w = V3.zero                          // spin, rad/s

    init(at p: V3) { self.p = p }
}

/// What the ball touched in a step.
enum Contact: Equatable {
    case table                               // the playing surface (which half: the ball's z)
    case net
    case edge                                // the table's side or edge
    case post
    case floor
    case paddle
}

/// Material of a contact: normal restitution and Coulomb friction (which turns sliding into spin).
struct Surface {
    let restitution: Double
    let friction: Double

    static let table = Surface(restitution: 0.905, friction: 0.25)
    static let rubber = Surface(restitution: 0.85, friction: 0.45)
    static let net = Surface(restitution: 0.15, friction: 0.3)
    static let post = Surface(restitution: 0.5, friction: 0.2)
    static let floor = Surface(restitution: 0.75, friction: 0.4)
    static let wood = Surface(restitution: 0.6, friction: 0.2)
}

/// Ball flight and contacts, in fixed substeps.
enum Physics {
    static let dt = 0.0005
    static let gravity = V3(0, -9.81, 0)
    /// Air drag a = -k |v| v: ½ ρ Cd A / m with ρ 1.2 kg/m³, Cd 0.5.
    static let dragK = 0.5 * 1.2 * 0.5 * Double.pi * Ball.radius * Ball.radius / Ball.mass
    /// Magnus a = c (ω × v): ½ ρ A r / m (lift coefficient ~ spin ratio).
    static let magnusK = 0.5 * 1.2 * Double.pi * Ball.radius * Ball.radius * Ball.radius / Ball.mass
    static let spinDecay = 0.25              // 1/s

    /// Free flight over one substep (semi-implicit Euler).
    static func fly(_ b: inout Ball, dt: Double = dt) {
        let speed = vlength(b.v)
        let a = gravity - b.v * (dragK * speed) + vcross(b.w, b.v) * magnusK
        b.v += a * dt
        b.p += b.v * dt
        b.w *= 1 - spinDecay * dt
    }

    /// Collision with the static scene (table, net, posts, floor) after a substep. Returns the contact
    /// and the normal impact speed if the ball bounced.
    static func collideStatic(_ b: inout Ball) -> (Contact, Double)? {
        let r = Ball.radius
        // Cheap reject: far from the table and above the floor.
        if b.p.y > r + 0.25, b.p.y - r > Table.floorY { return nil }
        if b.p.y - r <= Table.floorY {
            let n = V3(0, 1, 0)
            b.p.y = Table.floorY + r
            if let speed = resolve(&b, normal: n, surfaceVelocity: .zero, .floor) { return (.floor, speed) }
            return nil
        }
        let top = (min: V3(-Table.halfWidth, -Table.thickness, -Table.halfLength),
                   max: V3(Table.halfWidth, 0, Table.halfLength))
        if let (n, depth) = sphereBox(b.p, top.min, top.max) {
            b.p += n * depth
            if let speed = resolve(&b, normal: n, surfaceVelocity: .zero, .table) {
                return (n.y > 0.7 ? .table : .edge, speed)
            }
            return nil
        }
        let netBox = (min: V3(-Table.netHalfSpan, 0, -0.002), max: V3(Table.netHalfSpan, Table.netHeight, 0.002))
        if let (n, depth) = sphereBox(b.p, netBox.min, netBox.max) {
            b.p += n * depth
            if let speed = resolve(&b, normal: n, surfaceVelocity: .zero, .net) { return (.net, speed) }
            return nil
        }
        for side in [-1.0, 1.0] {
            let x = side * Table.netHalfSpan
            if let (n, depth) = sphereBox(b.p, V3(x - 0.012, -0.03, -0.012), V3(x + 0.012, Table.netHeight + 0.012, 0.012)) {
                b.p += n * depth
                if let speed = resolve(&b, normal: n, surfaceVelocity: .zero, .post) { return (.post, speed) }
                return nil
            }
        }
        return nil
    }

    /// Collision with a paddle blade (an elliptic disc), moving with `paddle`'s velocity. `grow`
    /// (assist, 0...1) enlarges the hit zone: up to 1.5x the blade and 3 cm in front of and behind it.
    static func collide(_ b: inout Ball, paddle: PaddleState, grow: Double = 0) -> Double? {
        let r = Ball.radius
        let rel = b.p - paddle.center
        guard vdot(rel, rel) < 0.2 * 0.2 else { return nil }
        let a = PaddleShape.semiWidth * (1 + 0.5 * grow), bb = PaddleShape.semiLength * (1 + 0.5 * grow)
        let h = PaddleShape.halfThickness + 0.03 * grow
        let l = paddle.basis.local(rel)
        var q = V3(l.x, l.y, clamped(l.z, -h, h))
        let e = (q.x / a) * (q.x / a) + (q.y / bb) * (q.y / bb)
        if e > 1 {
            let s = 1 / e.squareRoot()
            q.x *= s
            q.y *= s
        }
        let d = l - q
        let dist = vlength(d)
        guard dist < r else { return nil }
        let nl = dist > 1e-9 ? d / dist : V3(0, 0, l.z >= 0 ? 1 : -1)
        let n = paddle.basis.apply(nl)
        let contact = paddle.center + paddle.basis.apply(q)
        b.p += n * (r - dist)
        let u = paddle.velocity + vcross(paddle.angular, contact - paddle.center)
        return resolve(&b, normal: n, surfaceVelocity: u, .rubber)
    }

    /// Impulse at a contact with normal `n` (pointing to the ball) on a surface moving at `u`.
    /// Returns the normal approach speed, or nil if the ball was already separating.
    @discardableResult
    static func resolve(_ b: inout Ball, normal n: V3, surfaceVelocity u: V3, _ s: Surface) -> Double? {
        let r = Ball.radius
        // Velocity of the ball's contact point relative to the surface.
        let vc = b.v + vcross(b.w, n * -r) - u
        let vn = vdot(vc, n)
        guard vn < 0 else { return nil }
        // Slow contacts (rolling, resting) don't bounce.
        let restitution = -vn < 0.08 ? 0 : s.restitution
        let jn = -(1 + restitution) * vn
        b.v += n * jn
        // Friction: oppose sliding, at most until the contact point rolls (stick), at most μ·jn.
        let vt = vc - n * vn
        let slide = vlength(vt)
        if slide > 1e-9 {
            let t = vt / slide
            let jt = min(slide / (1 + 1 / Ball.inertia), s.friction * jn)
            b.v -= t * jt
            b.w += vcross(n, t) * (jt / (Ball.inertia * r))
        }
        if restitution == 0 {
            // Rolling on the surface: a little rolling resistance so it comes to rest.
            let tangential = b.v - n * vdot(b.v, n)
            b.v -= tangential * min(1, 0.6 * dt)
        }
        return -vn
    }

    /// Penetration of a sphere of radius r at `p` into the box: outward normal and depth.
    static func sphereBox(_ p: V3, _ lo: V3, _ hi: V3) -> (V3, Double)? {
        let r = Ball.radius
        let q = V3(clamped(p.x, lo.x, hi.x), clamped(p.y, lo.y, hi.y), clamped(p.z, lo.z, hi.z))
        let d = p - q
        let dist2 = vdot(d, d)
        guard dist2 < r * r else { return nil }
        if dist2 > 1e-18 {
            let dist = dist2.squareRoot()
            return (d / dist, r - dist)
        }
        // Centre inside the box: out through the nearest face.
        let faces: [(V3, Double)] = [
            (V3(1, 0, 0), hi.x - p.x), (V3(-1, 0, 0), p.x - lo.x),
            (V3(0, 1, 0), hi.y - p.y), (V3(0, -1, 0), p.y - lo.y),
            (V3(0, 0, 1), hi.z - p.z), (V3(0, 0, -1), p.z - lo.z),
        ]
        let nearest = faces.min { $0.1 < $1.1 }!
        return (nearest.0, nearest.1 + r)
    }
}
