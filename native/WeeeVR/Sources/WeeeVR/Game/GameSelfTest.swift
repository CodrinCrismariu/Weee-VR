import Foundation

/// Physics and opponent checks, logged at launch and run on Linux by scripts/game-test.
enum GameSelfTest {
    static func run() -> [String] {
        var out: [String] = []
        func cm(_ x: Double) -> String { String(format: "%.1f", x * 100) }

        // 1. ITTF ball spec: dropped from 30.5 cm onto the table it rebounds 23-25 cm.
        var b = Ball(at: V3(0, 0.305 + Ball.radius, 0.6))
        var bounced = false, apex = 0.0
        for _ in 0..<4000 {
            Physics.fly(&b)
            if Physics.collideStatic(&b) != nil { bounced = true }
            if bounced {
                apex = max(apex, b.p.y - Ball.radius)
                if b.v.y < 0, apex > 0.01 { break }
            }
        }
        out.append("drop 30.5 cm: rebound \(cm(apex)) cm (ITTF 23-25)")

        // 2. A paddle moving at 3 m/s into a ball at rest.
        var hitBall = Ball(at: V3(0, 0.3, 0.5))
        var paddle = PaddleState(center: hitBall.p + V3(0, 0, Ball.radius + PaddleShape.halfThickness - 0.002),
                                 basis: Basis.make(y: V3(0, 1, 0), z: V3(0, 0, -1)))
        paddle.velocity = V3(0, 0, -3)
        paddle.visible = true
        _ = Physics.collide(&hitBall, paddle: paddle)
        out.append(String(format: "paddle 3 m/s into a still ball: ball %.2f m/s (expect %.2f)", vlength(hitBall.v),
                          3 * (1 + Surface.rubber.restitution)))

        // 3. The opponent's aim: from behind its end line to a spot on the player's half.
        let from = V3(0.1, 0.25, -Table.halfLength - 0.1), target = V3(0.3, Ball.radius, 0.9)
        if let (v, w) = Opponent.aim(from: from, to: target, speed: 7, topspin: 100) {
            var s = Ball(at: from)
            s.v = v
            s.w = w
            var clearance = -1.0, landing: V3?
            for _ in 0..<6000 {
                let z = s.p.z
                Physics.fly(&s)
                if z < 0, s.p.z >= 0 { clearance = s.p.y - Ball.radius - Table.netHeight }
                if let (c, _) = Physics.collideStatic(&s) {
                    if c == .table { landing = s.p }
                    break
                }
            }
            if let landing {
                out.append("opponent 7 m/s topspin: lands \(cm(vlength(V3(landing.x - target.x, 0, landing.z - target.z)))) cm from target, net clearance \(cm(clearance)) cm, launch \(String(format: "%.1f", asin(v.y / vlength(v)) * 180 / .pi)) deg")
            } else {
                out.append("FAIL opponent shot did not land on the table")
            }
        } else {
            out.append("FAIL opponent could not aim")
        }

        // 4. A whole exchange through the game loop: a good player shot, the opponent returns it,
        //    nobody hits it back.
        let game = PingPong()
        var events: [String] = []
        var sounds = [Int](repeating: 0, count: GameSound.allCases.count)
        game.onEvent = { events.append($0) }
        game.onSound = { kind, _, _ in sounds[kind.rawValue] += 1 }
        game.placeForTest()
        game.opponent.misses = false
        // The player's shot: aimed like the opponent's, mirrored to go from the player's side.
        let mirrored = { (v: V3) in V3(v.x, v.y, -v.z) }
        let playerFrom = V3(-0.1, 0.3, Table.halfLength + 0.15)
        if let (v, w) = Opponent.aim(from: mirrored(playerFrom), to: V3(0.2, Ball.radius, 0.7), speed: 6, topspin: 80) {
            var shot = Ball(at: playerFrom)
            shot.v = mirrored(v)
            shot.w = V3(-w.x, -w.y, w.z)                        // spin is a pseudovector
            game.injectForTest(shot)
            var t = 0.0
            while t < 4 {
                t += 1.0 / 60
                game.update(time: t, paddle: PaddleState(), pinch: nil)
                game.opponent.waitForPlan()
            }
            out.append("exchange: opponent returns \(game.best) (expect 1), events \(events), sounds paddle/table/net/floor \(sounds)")
        } else {
            out.append("FAIL player test shot")
        }

        // 5. Full assist: a sloppy serve (paddle face tilted up and sideways) at the floating ball
        //    still lands on the opponent's half and gets returned.
        let assisted = PingPong()
        assisted.settings.assist = 1
        assisted.settings.floatingServe = true
        var landings: [String] = []
        assisted.onSound = { kind, p, _ in if kind == .table { landings.append(String(format: "z %.2f", p.z)) } }
        assisted.placeForTest()
        assisted.opponent.misses = false
        var t = 0.0
        while t < 3 {
            t += 1.0 / 60
            let serve = PingPong.servePoint
            var pose = PaddleState(center: V3(serve.x + 0.03, serve.y, serve.z + 0.25 - 2.5 * max(t - 0.2, 0)),
                                   basis: Basis.make(y: V3(0, 1, 0), z: vnormalize(V3(0.3, 0.9, -1))))
            pose.visible = t < 0.6
            assisted.update(time: t, paddle: pose, pinch: nil)
            assisted.opponent.waitForPlan()
        }
        out.append("assist 1, sloppy serve: table contacts \(landings) (first z < 0 = on the opponent's half), opponent returns \(assisted.best) (expect 1)")
        return out
    }
}

extension PingPong {
    /// Self-test: the table at the world origin.
    func placeForTest() { place(head: SIMD3<Float>(0, Float(settings.tableDrop), Float(0.45 + Table.halfLength)), yaw: 0) }
}
