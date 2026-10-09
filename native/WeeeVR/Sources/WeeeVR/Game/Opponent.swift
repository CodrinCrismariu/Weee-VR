import Foundation

/// The automatic opponent at the far end. When the player's shot crosses the net it predicts the
/// ball (the same physics, so the prediction is exact), picks where to meet it (top of the bounce),
/// and solves a return: a landing spot on the player's half, a speed and topspin, the launch angle
/// found by simulating the flight (bisection), clearing the net. Its paddle then swings through the
/// contact point at the contact moment; at that instant the ball leaves with the solved velocity.
/// Solving runs on a utility queue (a few ms), well before the contact.
final class Opponent {
    struct Shot {
        var contactTime: Double
        var contactPoint: V3
        var velocity: V3
        var spin: V3
        var miss: Bool
    }

    /// 0 (gentle) ... 1 (fast, rarely misses).
    var level = 0.5
    /// Occasionally swing late on purpose (off for the self-test).
    var misses = true

    private(set) var paddle = Opponent.readyPose
    private var shot: Shot?
    private var applied = false
    private var approachStart: (time: Double, pose: PaddleState)?
    private var shotID = 0                    // a plan for an older ball is ignored
    private var seeds = GameRandom(seed: 0x51_1EED)

    private let queue = DispatchQueue(label: "game.opponent", qos: .utility)
    private let lock = NSLock()
    private var finished: (id: Int, shot: Shot?)?

    static let readyPose: PaddleState = {
        var p = PaddleState(center: V3(0.1, 0.28, -Table.halfLength - 0.45),
                            basis: Basis.make(y: V3(-0.25, 1, 0), z: V3(0, 0, 1)))
        p.visible = true
        return p
    }()

    func reset() {
        shotID += 1
        shot = nil
        approachStart = nil
        paddle = Self.readyPose
    }

    /// Render thread: the player's ball, just past the net at `time`. Solving happens off-thread.
    /// `assist` (0...1) draws the return toward `forehand` (the player's paddle x) at a medium depth.
    func plan(ball: Ball, time: Double, assist: Double = 0, forehand: Double = 0.2) {
        shotID += 1
        let id = shotID, level = self.level, misses = self.misses
        let seed = UInt64(seeds.next() * 1e15)
        queue.async { [self] in
            let shot = Self.solve(ball, at: time, level: level, misses: misses, assist: assist, forehand: forehand,
                                  seed: seed)
            lock.lock()
            finished = (id, shot)
            lock.unlock()
        }
    }

    /// Tests: wait for a pending plan.
    func waitForPlan() { queue.sync {} }

    /// Render thread, once per frame: take a finished plan and move the paddle.
    func update(time: Double) {
        lock.lock()
        let done = finished
        finished = nil
        lock.unlock()
        if let done, done.id == shotID, let s = done.shot, s.contactTime > time + 0.05 {
            shot = s
            applied = false
            approachStart = (time, paddle)
        }
        animate(time)
    }

    /// Physics substep at `time`: if this is the planned contact (and the ball arrived as predicted),
    /// the ball leaves with the planned velocity. True when it was hit.
    func hit(_ b: inout Ball, at time: Double) -> Bool {
        guard let s = shot, !applied, time >= s.contactTime - Physics.dt / 2 else { return false }
        applied = true
        guard !s.miss, time < s.contactTime + Physics.dt, vlength(b.p - s.contactPoint) < 0.03 else { return false }
        b.p = s.contactPoint
        b.v = s.velocity
        b.w = s.spin
        return true
    }

    // MARK: Swing

    private func animate(_ time: Double) {
        guard let s = shot, let start = approachStart else {
            paddle = Self.readyPose
            return
        }
        let up = V3(0, 1, 0)
        let direction = vnormalize(s.velocity == .zero ? V3(0, 0.2, 1) : s.velocity)
        // Contact pose: the forehand face behind the ball, closed a little (topspin), handle down-left.
        let face = vnormalize(direction - up * 0.25)
        let contactBasis = Basis.make(y: V3(-0.4, 1, 0), z: face)
        let contact = s.contactPoint - face * (Ball.radius + PaddleShape.halfThickness)
        let late = s.miss ? 0.09 : 0                      // a miss: the swing comes too late
        let tc = s.contactTime + late
        let tb = tc - 0.16, tf = tc + 0.14, tr = tf + 0.6
        let back = contact - direction * 0.30 - up * 0.12
        let through = contact + direction * 0.25 + up * 0.22
        let swing = direction * 3.0 + up * 1.0
        var pose = PaddleState(center: contact, basis: contactBasis)
        pose.visible = true
        if time < tb {
            let f = minimumJerk((time - start.time) / max(tb - start.time, 0.05))
            pose.center = start.pose.center + (back - start.pose.center) * f
            pose.basis = start.pose.basis.rotated(by: start.pose.basis.rotationVector(to: contactBasis) * f)
        } else if time < tc {
            pose.center = hermite(back, .zero, contact, swing, duration: tc - tb, at: time - tb)
        } else if time < tf {
            pose.center = hermite(contact, swing, through, .zero, duration: tf - tc, at: time - tc)
        } else if time < tr {
            let f = minimumJerk((time - tf) / (tr - tf))
            pose.center = through + (Self.readyPose.center - through) * f
            pose.basis = contactBasis.rotated(by: contactBasis.rotationVector(to: Self.readyPose.basis) * f)
        } else {
            shot = nil
            approachStart = nil
            pose = Self.readyPose
        }
        paddle = pose
    }

    // MARK: Planning (utility queue)

    /// The return of `start` (the ball's state at `time`), or nil if the ball won't land legally on
    /// the opponent's half.
    static func solve(_ start: Ball, at time: Double, level: Double, misses: Bool, assist: Double = 0,
                      forehand: Double = 0.2, seed: UInt64) -> Shot? {
        guard let (steps, ball) = predictContact(start) else { return nil }
        var rng = GameRandom(seed: seed)
        let x = rng.uniform(-0.5, 0.5), depth = rng.uniform(0.35, 1.1)
        let target = V3(x + (forehand - x) * assist, Ball.radius, Table.halfLength - (depth + (0.75 - depth) * assist))
        var speed = 4 + 5 * level * rng.uniform(0.75, 1.0)
        let spin = rng.uniform(60, 150) * (0.6 + 0.4 * level)
        let miss = misses && rng.next() < 0.15 - 0.12 * level
        let contactTime = time + Double(steps) * Physics.dt
        while speed >= 2.5 {
            if let (v, w) = aim(from: ball.p, to: target, speed: speed, topspin: spin) {
                return Shot(contactTime: contactTime, contactPoint: ball.p, velocity: v, spin: w, miss: miss)
            }
            speed *= 0.88
        }
        return Shot(contactTime: contactTime, contactPoint: ball.p, velocity: .zero, spin: .zero, miss: true)
    }

    /// Where the opponent meets the ball: its flight from `start` (static scene only, the same
    /// substeps the game runs) must bounce once on the opponent's half; contact at the top of the
    /// bounce, or when it gets behind the end line. (substeps from `start`, ball state then)
    static func predictContact(_ start: Ball) -> (Int, Ball)? {
        var b = start
        var bounced = false
        for step in 1...6000 {
            Physics.fly(&b)
            if let (contact, _) = Physics.collideStatic(&b) {
                guard contact == .table, b.p.z < 0, !bounced else { return nil }
                bounced = true
            }
            if bounced, b.v.y <= 0 || b.p.z < -Table.halfLength - 0.25 {
                return b.p.y > 0.04 ? (step, b) : nil
            }
        }
        return nil
    }

    /// Launch velocity and spin from `p` landing at `target` with `speed` and `topspin`, clearing the
    /// net by 3 cm; nil if no angle works at this speed. A coarser `dt` and fewer bisection
    /// `iterations` make it cheap enough for the render thread (the player's assisted shots).
    static func aim(from p: V3, to target: V3, speed: Double, topspin: Double, dt: Double = Physics.dt,
                    iterations: Int = 24) -> (V3, V3)? {
        let flat = V3(target.x - p.x, 0, target.z - p.z)
        let distance = vlength(flat)
        guard distance > 0.3 else { return nil }
        let direction = flat / distance
        let up = V3(0, 1, 0)
        let spin = vnormalize(vcross(up, direction)) * topspin     // top of the ball rolls forward

        // Landing error along the shot (negative: short / into the net) and net clearance.
        func flight(_ angle: Double) -> (error: Double, clearance: Double) {
            var b = Ball(at: p)
            b.v = (direction * cos(angle) + up * sin(angle)) * speed
            b.w = spin
            var clearance = -1.0
            var previousZ = b.p.z
            for _ in 0..<Int(3 / dt) {
                Physics.fly(&b, dt: dt)
                if previousZ < 0, b.p.z >= 0 { clearance = b.p.y - Ball.radius - Table.netHeight }
                previousZ = b.p.z
                guard let (contact, _) = Physics.collideStatic(&b) else { continue }
                switch contact {
                case .table:
                    if b.p.z < 0 { return (-10, clearance) }
                    return (vdot(V3(b.p.x - p.x, 0, b.p.z - p.z), direction) - distance, clearance)
                case .net, .post: return (-10, clearance)
                case .edge, .floor, .paddle: return (b.p.z > 0 ? 10 : -10, clearance)
                }
            }
            return (10, clearance)
        }

        var lo = -0.35, hi = 0.75
        guard flight(lo).error < 0, flight(hi).error > 0 else { return nil }
        for _ in 0..<iterations {
            let mid = (lo + hi) / 2
            if flight(mid).error < 0 { lo = mid } else { hi = mid }
        }
        let angle = (lo + hi) / 2
        let result = flight(angle)
        guard abs(result.error) < 0.05, result.clearance >= 0.03 else { return nil }
        return ((direction * cos(angle) + up * sin(angle)) * speed, spin)
    }
}
