import Foundation

enum GameSound: Int, CaseIterable {
    case paddle, table, net, floor
}

struct GameSettings {
    var tableDrop = 0.80                     // table surface below the eyes when placed, m
    var opponentLevel = 0.5
    var gripAngle = 40.0 * Double.pi / 180   // blade axis from the knuckle direction toward the thumb
    var gripOffset = -0.012                  // blade plane along the palm normal, m
    /// 0 (realistic) ... 1: a more forgiving paddle, shots steered onto the table, a slower ball,
    /// a clean toss and friendlier returns (see `PingPong`).
    var assist = 0.0
    var floatingServe = false                // a new ball hovers at serve height instead of resting
}

/// Table tennis against the automatic opponent, in the table frame (`Table`).
///
/// - The ball rests on the player's half until the left hand pinches it (thumb and index tips
///   together near it); opening the pinch releases it with the hand's velocity, so it can be tossed.
/// - The right hand holds the paddle (`PaddleState.grip`). Physics runs in 0.5 ms substeps up to each
///   frame's display time, with the paddle swept between the frame poses, so fast swings hit.
/// - Practice rules: a player shot must land on the opponent's half (a serve may bounce on the
///   player's side first); the opponent returns those; a return must bounce once on the player's half.
///   Anything else ends the rally. A ball that reaches the floor is replaced at once by a new one on the
///   table; otherwise a new ball appears after a moment.
/// - Assist (`GameSettings.assist`, 0...1), for tracking that is only so precise and so fast:
///   the paddle's hit zone grows (up to 1.5x wider, 3 cm in front of and behind the blade); the
///   player's shot is blended toward one that clears the net and lands on the opponent's half,
///   keeping its direction, pace and spin; the game runs slower (down to 0.65x); a released ball's
///   throw is straightened toward a soft vertical toss; the opponent aims nearer the player's forehand.
///   `floatingServe`: a new ball hovers at serve height, waiting to be hit.
final class PingPong {
    enum Phase: Equatable {
        case ready                          // resting at the spawn point
        case held
        case inPlay
        case over(until: Double)            // rally ended; physics continues until the respawn
    }

    enum Hitter {
        case none, player, opponent
    }

    var settings = GameSettings() {
        didSet { opponent.level = settings.opponentLevel }
    }
    /// A sound at a world position, with the impact speed (m/s).
    var onSound: ((GameSound, SIMD3<Float>, Double) -> Void)?
    /// Rally events, for the log.
    var onEvent: ((String) -> Void)?

    private(set) var placed = false
    private(set) var origin = V3.zero        // table frame origin in the world
    private(set) var yaw = 0.0               // table frame heading in the world (about +y)
    private(set) var ball = Ball(at: Table.spawn)
    private(set) var phase = Phase.ready
    private(set) var paddle = PaddleState()
    private(set) var rally = 0
    private(set) var best = 0
    let opponent = Opponent()

    private var realTime: Double?            // the last frame's (display) time
    private var clock = 0.0                  // game time: real time, slowed by the assist
    private var simTime: Double?             // game time of `ball`'s state
    private var lastHitter = Hitter.none
    private var serve = false
    private var bouncesPlayer = 0, bouncesOpponent = 0
    private var lastPaddleContact = -1.0
    private var pinchClosed = false
    private var pinchTrail: [(time: Double, point: V3)] = []
    private var leftMissingSince: Double?
    private var restingSince: Double?

    // MARK: Placement

    /// The table ahead of the eyes at `head` (world), its length along `yaw` (the gaze heading).
    func place(head: SIMD3<Float>, yaw: Double) {
        self.yaw = yaw
        let forward = V3(-sin(yaw), 0, -cos(yaw))
        let eyes = V3(Double(head.x), Double(head.y), Double(head.z))
        origin = eyes + forward * (0.45 + Table.halfLength)
        origin.y = eyes.y - settings.tableDrop
        placed = true
        respawn()
        onEvent?("table placed")
    }

    func toTable(_ p: SIMD3<Float>) -> V3 {
        let d = V3(Double(p.x), Double(p.y), Double(p.z)) - origin
        let c = cos(yaw), s = sin(yaw)
        return V3(c * d.x - s * d.z, d.y, s * d.x + c * d.z)
    }

    func toWorld(_ l: V3) -> SIMD3<Float> {
        let c = cos(yaw), s = sin(yaw)
        let w = origin + V3(c * l.x + s * l.z, l.y, -s * l.x + c * l.z)
        return SIMD3<Float>(Float(w.x), Float(w.y), Float(w.z))
    }

    // MARK: Frame update

    /// Advance to `time` (the frame's display time) with the hands' joints there (world, MediaPipe
    /// order; nil when not tracked).
    func update(time: Double, right: [SIMD3<Float>]?, left: [SIMD3<Float>]?) {
        guard placed else { return }
        var target = PaddleState()
        if let right, right.count >= 21,
           let grip = PaddleState.grip(right.map(toTable), gripAngle: settings.gripAngle, offset: settings.gripOffset) {
            target = grip
        }
        var pinch: (V3, V3)?
        if let left, left.count >= 21 { pinch = (toTable(left[4]), toTable(left[8])) }
        update(time: time, paddle: target, pinch: pinch)
    }

    /// Game time runs this much slower than real time (assist).
    private var timeScale: Double { 1 - 0.35 * settings.assist }

    /// The same in the table frame (also the self-test's entry point).
    func update(time: Double, paddle target: PaddleState, pinch: (thumb: V3, index: V3)?) {
        let realElapsed = realTime.map { min(max(time - $0, 0), 0.1) } ?? 0     // a stall isn't simulated
        realTime = time
        let from = clock
        clock += realElapsed * timeScale
        let span = clock - from
        let previous = paddle
        opponent.update(time: clock)
        updatePinch(pinch, time: time)
        if phase == .ready, settings.floatingServe { ball.p = Self.servePoint + V3(0, 0.008 * sin(clock * 2.5), 0) }

        var t = max(simTime ?? clock, from - Physics.dt)
        while t + Physics.dt <= clock + 1e-9 {
            t += Physics.dt
            var pose = target
            if previous.visible, target.visible, span > 0 {
                pose = previous.interpolated(to: target, fraction: clamped((t - from) / span, 0, 1), over: span)
            }
            step(at: t, paddle: pose)
        }
        simTime = t
        if previous.visible, target.visible, span > 0 {
            paddle = previous.interpolated(to: target, fraction: 1, over: span)
        } else {
            paddle = target
        }
        if case .over(let until) = phase, clock >= until { respawn() }
    }

    /// Floating serve: above the player's end of the table, a little to the forehand side.
    static let servePoint = V3(0.15, 0.30, Table.halfLength - 0.12)

    /// Whether the paddle may touch the ball at `t`: with the assist, not again right after a hit
    /// (the steered ball must not be caught by the same swing).
    private func paddleCanHit(at t: Double) -> Bool {
        settings.assist == 0 || t - lastPaddleContact > 0.1
    }

    private func step(at t: Double, paddle pose: PaddleState) {
        switch phase {
        case .held:
            return
        case .ready:
            guard pose.visible, let speed = Physics.collide(&ball, paddle: pose, grow: settings.assist) else { return }
            phase = .inPlay
            lastHitter = .none
            playerHit(speed: speed, at: t)
            return
        case .inPlay, .over:
            break
        }
        let previousZ = ball.p.z
        Physics.fly(&ball)
        if let (contact, speed) = Physics.collideStatic(&ball) { touched(contact, speed: speed, at: t) }
        if pose.visible, paddleCanHit(at: t), let speed = Physics.collide(&ball, paddle: pose, grow: settings.assist) {
            playerHit(speed: speed, at: t)
        }
        if opponent.hit(&ball, at: t) { opponentHit(at: t) }
        guard phase == .inPlay else { return }
        if lastHitter == .player, previousZ >= 0, ball.p.z < 0 {
            opponent.plan(ball: ball, time: t, assist: settings.assist,
                          forehand: clamped(paddle.visible ? paddle.center.x : 0.2, -0.5, 0.5))
        }
        // Out of play: at rest, or far away.
        if vlength(ball.v) < 0.05 {
            if restingSince == nil { restingSince = t }
            if t - restingSince! > 1 { endRally("ball stopped", at: t) }
        } else {
            restingSince = nil
        }
        if abs(ball.p.x) > 6 || abs(ball.p.z) > 8 { endRally("ball gone", at: t) }
    }

    // MARK: Rules

    private func touched(_ contact: Contact, speed: Double, at t: Double) {
        switch contact {
        case .table, .edge: sound(.table, speed)
        case .net, .post: sound(.net, speed)
        case .floor: sound(.floor, speed)
        case .paddle: break
        }
        if contact == .floor {
            // Off the table and on the ground: the rally (if any) is over, and a new ball is ready at once.
            if phase == .inPlay, lastHitter != .none {
                endRally(lastHitter == .opponent && bouncesPlayer == 0 ? "opponent out" : "floor", at: t)
            }
            respawn()
            return
        }
        guard phase == .inPlay else { return }
        switch contact {
        case .table:
            let playerSide = ball.p.z > 0
            switch lastHitter {
            case .player:
                if playerSide {
                    if serve, bouncesPlayer == 0, bouncesOpponent == 0 { bouncesPlayer = 1 } else { endRally("bounced on your side", at: t) }
                } else if bouncesOpponent == 0 {
                    bouncesOpponent = 1
                } else {
                    endRally("opponent missed", at: t)
                }
            case .opponent:
                if playerSide, bouncesPlayer == 0 {
                    bouncesPlayer = 1
                } else {
                    endRally(playerSide ? "you missed" : "opponent fault", at: t)
                }
            case .none:
                break                                   // a tossed or dropped ball
            }
        case .edge:
            if lastHitter != .none { endRally("edge", at: t) }
        case .floor, .net, .post, .paddle:
            break
        }
    }

    private func playerHit(speed: Double, at t: Double) {
        let fresh = t - lastPaddleContact > 0.08          // one hit, not every substep of the contact
        lastPaddleContact = t
        guard fresh else { return }
        sound(.paddle, speed)
        steerShot()
        if phase == .inPlay {
            serve = lastHitter == .none
            lastHitter = .player
            bouncesPlayer = 0
            bouncesOpponent = 0
            opponent.reset()
        }
    }

    /// Assist: blend the ball's flight off the paddle toward one that clears the net and lands on the
    /// opponent's half, with the same direction (aimed at the table if it wasn't), pace (3-8 m/s, a
    /// faster shot lands deeper) and topspin or backspin. Solved like the opponent's shots (coarser
    /// steps, well under a millisecond).
    private func steerShot() {
        let a = settings.assist
        guard a > 0 else { return }
        let v = ball.v
        var direction = V3(v.x, 0, v.z)
        if vlength(direction) < 0.3 || direction.z > -0.3 * vlength(direction) { direction = V3(0, 0, -1) }
        direction = vnormalize(direction)
        let pace = clamped(vlength(v), 3, 8)
        let depth = clamped(0.45 + (pace - 3) * 0.12, 0.45, 1.05)          // landing distance from the net
        let x = clamped(ball.p.x + direction.x / -direction.z * (ball.p.z + depth), -0.6, 0.6)
        let spinAxis = vnormalize(vcross(V3(0, 1, 0), direction))
        let spin = clamped(vdot(ball.w, spinAxis), -80, 150)
        // Opponent.aim shoots toward +z: solve mirrored (z -> -z; spin, a pseudovector: x, y flip).
        let from = V3(ball.p.x, ball.p.y, -ball.p.z)
        for speed in [pace, pace * 0.8, 3.0] {
            guard let (mv, mw) = Opponent.aim(from: from, to: V3(x, Ball.radius, depth), speed: speed, topspin: spin,
                                              dt: 0.002, iterations: 14) else { continue }
            ball.v = v + (V3(mv.x, mv.y, -mv.z) - v) * a
            ball.w = ball.w + (V3(-mw.x, -mw.y, mw.z) - ball.w) * a
            return
        }
    }

    private func opponentHit(at t: Double) {
        sound(.paddle, vlength(ball.v))
        lastHitter = .opponent
        serve = false
        bouncesPlayer = 0
        bouncesOpponent = 0
        rally += 1
        best = max(best, rally)
    }

    private func endRally(_ reason: String, at t: Double) {
        guard phase == .inPlay else { return }
        phase = .over(until: t + 1.5)
        onEvent?("rally over: \(reason), rally \(rally), best \(best)")
    }

    private func respawn() {
        ball = Ball(at: settings.floatingServe ? Self.servePoint : Table.spawn)
        phase = .ready
        lastHitter = .none
        serve = false
        bouncesPlayer = 0
        bouncesOpponent = 0
        rally = 0
        restingSince = nil
        opponent.reset()
    }

    /// Self-test: `b` as a ball the player just hit.
    func injectForTest(_ b: Ball) {
        ball = b
        phase = .inPlay
        lastHitter = .player
        serve = false
        bouncesPlayer = 0
        bouncesOpponent = 0
    }

    // MARK: Left hand

    private func updatePinch(_ pinch: (thumb: V3, index: V3)?, time: Double) {
        guard let pinch else {
            pinchClosed = false
            pinchTrail.removeAll()
            if phase == .held {
                if leftMissingSince == nil { leftMissingSince = time }
                if time - leftMissingSince! > 0.25 { release(at: time, velocity: .zero) }
            }
            return
        }
        leftMissingSince = nil
        let gap = vlength(pinch.thumb - pinch.index)
        let point = (pinch.thumb + pinch.index) / 2
        pinchTrail.append((time, point))
        pinchTrail.removeAll { time - $0.time > 0.08 }

        if pinchClosed {
            if gap > 0.05 {
                pinchClosed = false
                if phase == .held {
                    // Assist: the throw straightened toward a soft vertical toss. Game time may run
                    // slower than real time: the ball leaves the hand at the hand's real speed.
                    let thrown = pinchVelocity()
                    let toss = V3(0, clamped(thrown.y, 1.4, 2.4), 0)
                    release(at: time, velocity: (thrown + (toss - thrown) * settings.assist) / timeScale)
                }
            }
        } else if gap < 0.035 {
            pinchClosed = true
            if phase != .held, vlength(point - ball.p) < 0.08 {
                phase = .held
                lastHitter = .none
                rally = 0
                opponent.reset()
            }
        }
        if phase == .held {
            ball.p = point
            ball.v = pinchVelocity()
            ball.w = .zero
        }
    }

    private func release(at time: Double, velocity: V3) {
        phase = .inPlay
        lastHitter = .none
        serve = false
        bouncesPlayer = 0
        bouncesOpponent = 0
        ball.v = velocity
        restingSince = nil
    }

    /// Velocity of the pinch point over the last ~80 ms (least squares), m/s.
    private func pinchVelocity() -> V3 {
        guard pinchTrail.count >= 2 else { return .zero }
        let t0 = pinchTrail[0].time
        let n = Double(pinchTrail.count)
        var st = 0.0, stt = 0.0, sp = V3.zero, stp = V3.zero
        for (time, point) in pinchTrail {
            let t = time - t0
            st += t
            stt += t * t
            sp += point
            stp += point * t
        }
        let denominator = n * stt - st * st
        guard denominator > 1e-9 else { return .zero }
        let v = (stp * n - sp * st) / denominator
        return vlength(v) > 8 ? vnormalize(v) * 8 : v
    }

    private func sound(_ kind: GameSound, _ speed: Double) {
        onSound?(kind, toWorld(ball.p), speed)
    }

    // MARK: Renderer

    /// The scene for the shader (`ShaderSource`, buffer 2), table frame; `sceneCount` float4s.
    static let sceneCount = 12
    func sceneData() -> [SIMD4<Float>] {
        var out = [SIMD4<Float>](repeating: .zero, count: Self.sceneCount)
        guard placed else { return out }
        func f3(_ v: V3) -> SIMD3<Float> { SIMD3<Float>(Float(v.x), Float(v.y), Float(v.z)) }
        var flags: Float = 1                                   // ball
        if paddle.visible { flags += 2 }
        if opponent.paddle.visible { flags += 4 }
        out[0] = SIMD4(f3(origin), Float(yaw))
        out[1] = SIMD4(Float(cos(yaw)), Float(sin(yaw)), Float(Table.floorY), flags)
        out[2] = SIMD4(f3(ball.p), Float(Ball.radius))
        out[3] = SIMD4(f3(vnormalize(V3(0.25, 1.0, 0.35))), 0)
        for (i, p) in [paddle, opponent.paddle].enumerated() {
            let base = 4 + 4 * i
            out[base] = SIMD4(f3(p.center), p.visible ? 1 : 0)
            out[base + 1] = SIMD4(f3(p.basis.x), 0)
            out[base + 2] = SIMD4(f3(p.basis.y), 0)
            out[base + 3] = SIMD4(f3(p.basis.z), 0)
        }
        return out
    }

    /// One HUD/log line.
    var summary: String {
        let state: String
        switch phase {
        case .ready: state = "pick up the ball"
        case .held: state = "holding"
        case .inPlay: state = "in play"
        case .over: state = "point over"
        }
        return "pong \(state) · rally \(rally) · best \(best)"
    }
}
