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
        out.append(contentsOf: paddleFusion())
        return out
    }

    /// 6. The smart paddle: a simulated paddle turned in view for the calibration (unknown IMU mount,
    ///    tag turn and clock offset, a drifting IMU heading, noisy and sometimes ambiguous tag views),
    ///    then played with, with the tag out of view for 0.4 s; then turned over so only the two small
    ///    forehand tags are seen (their turns unknown at first).
    static func paddleFusion() -> [String] {
        var rng = GameRandom(seed: 11)
        func noise(_ scale: Double) -> V3 { V3(rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)) * scale }
        let mount = Quat(rotationVector: V3(0.4, -1.1, 2.0))              // IMU <- tag
        let turns = 1
        let tagFromPaddle = Quat(rotationVector: V3(0, 0, Double(turns) * .pi / 2))
        let heading = Quat(rotationVector: V3(-.pi / 2, 0, 0)) * Quat(rotationVector: V3(0, 0, 0.7))   // earth z-up -> world
        let drift = 0.5 * .pi / 180                                       // rad/s, IMU heading drift
        let timeOffset = 0.012                                            // IMU stamps 12 ms late
        let palmToGrip = V3(0.01, 0, -0.02)                               // paddle frame

        func paddle(_ t: Double) -> (rotation: Quat, grip: V3) {
            let r = Quat(rotationVector: V3(0, 0.3, 0)) * Quat(rotationVector: V3(0.9 * sin(1.3 * t), 0, 0))
                * Quat(rotationVector: V3(0, 0.8 * sin(0.9 * t + 1), 0)) * Quat(rotationVector: V3(0, 0, 0.7 * sin(1.7 * t + 2)))
            let g = V3(0.25 + 0.1 * sin(0.7 * t), 1.0 + 0.05 * sin(1.1 * t), -0.4 + 0.08 * sin(0.5 * t))
            return (r.normalized, g)
        }
        func imuOrientation(_ t: Double) -> Quat {
            let tag = paddle(t).rotation * tagFromPaddle.conjugate
            let b = heading.conjugate * tag * mount.conjugate
            return (Quat(rotationVector: V3(0, 0, drift * t)) * b).normalized
        }

        let fusion = PaddleFusion()
        var calibrated: PaddleCalibration?
        fusion.onCalibrated = { calibrated = $0 }
        fusion.beginCalibration()

        let dt = 1.0 / 600
        var nextIMU = 0.0, nextFrame = 0.0
        var errors: (pos: [Double], ang: [Double], gapPos: [Double], gapAng: [Double]) = ([], [], [], [])
        var smallErrors: (pos: [Double], ang: [Double]) = ([], [])
        let smallTurns = [68: 1, 69: 3]
        // A view of a tag (world <- tag) with noise and the planar ambiguity's other pose.
        func view(_ tag: Quat, at position: V3) -> [PaddleTagSighting.Candidate] {
            let seen = Quat(rotationVector: noise(0.006)) * tag
            // The planar ambiguity: the other pose has the tag's tilt mirrored about the line of
            // sight from the camera (at head height, 0.5 m behind), so it differs view by view.
            let sight = vnormalize(position - V3(0.1, 1.5, 0.1))
            let normal = seen.act(V3(0, 0, 1))
            let mirrored = sight * (2 * vdot(normal, sight)) - normal
            let axis = vcross(normal, mirrored)
            let turn = vlength(axis) > 1e-9 ? vnormalize(axis) * acos(clamped(vdot(normal, mirrored), -1, 1)) : .zero
            let flipped = Quat(rotationVector: turn) * seen
            var candidates = [PaddleTagSighting.Candidate(rotation: seen, position: position + noise(0.002), error: 1e-5),
                              PaddleTagSighting.Candidate(rotation: flipped, position: position, error: 5e-4)]
            if rng.next() < 0.3 {                       // ambiguous view, sometimes the wrong one first
                candidates[1].error = 1.2e-5
                if rng.next() < 0.5 { candidates.swapAt(0, 1) }
            }
            return candidates
        }
        var t = 0.0
        while t < 20 {
            t += dt
            // IMU samples reach the phone ~5 ms after they're taken.
            while nextIMU <= t - 0.005 {
                let b = imuOrientation(nextIMU)
                let rate = (b.conjugate * imuOrientation(nextIMU + 1e-4)).rotationVector / 1e-4
                fusion.add([PaddleIMUSample(time: nextIMU + timeOffset, orientation: b, rate: rate, still: false)])
                nextIMU += 0.005
            }
            fusion.tick(time: t)
            if t >= nextFrame {
                let (r, g) = paddle(nextFrame)
                let gap = nextFrame > 11 && nextFrame < 11.4
                fusion.addPalm(g - r.act(palmToGrip) + noise(0.015), time: nextFrame)
                let centre = g - r.act(PaddleFusion.gripFromCentre)
                if nextFrame < 14, !gap, rng.next() > 0.15 {
                    let tag = r * tagFromPaddle.conjugate
                    let position = centre - r.act(PaddleFusion.tagToCentre)
                    fusion.add(PaddleTagSighting(id: 0, time: nextFrame, candidates: view(tag, at: position)))
                } else if nextFrame >= 14 {
                    // The forehand side: the two small tags, each in most frames.
                    for (id, offset) in PaddleFusion.smallTags where rng.next() > 0.15 {
                        let tag = r * PaddleFusion.forehandPlacements[smallTurns[id]!].conjugate
                        fusion.add(PaddleTagSighting(id: id, time: nextFrame, candidates: view(tag, at: centre + r.act(offset))))
                    }
                }
                nextFrame += 1.0 / 30
            }
            // Played: the paddle at the display time, 20 ms ahead.
            if t > 9, Int(t / dt) % 10 == 0, let pose = fusion.pose(at: t + 0.02) {
                let (r, g) = paddle(t + 0.02)
                let centre = g - r.act(PaddleFusion.gripFromCentre)
                let pos = vlength(pose.center - centre)
                let ang = Quat.angle(Quat(basis: pose.basis), r)
                if t > 11, t < 11.5 { errors.gapPos.append(pos); errors.gapAng.append(ang) }
                else if t < 14 { errors.pos.append(pos); errors.ang.append(ang) }
                else if t > 16 { smallErrors.pos.append(pos); smallErrors.ang.append(ang) }
            }
        }

        var out: [String] = []
        guard let cal = calibrated else { return ["FAIL paddle fusion: calibration did not finish (\(fusion.summary))"] }
        let mountError = Quat.angle(cal.mountQuat, mount) * 180 / .pi
        out.append(String(format: "paddle calibration: mount off %.2f° (< 2), clock offset %.1f ms (true %.1f, within 3), tag turns %d (true %d), residual %.2f°",
                          mountError, cal.timeOffset * 1000, timeOffset * 1000, cal.tagTurns, turns, cal.residualDeg))
        func worst(_ a: [Double]) -> Double { a.max() ?? .nan }
        out.append(String(format: "paddle tracking: worst %.1f mm / %.2f° with the tag, %.1f mm / %.2f° in a 0.4 s tag gap (%d + %d poses)",
                          worst(errors.pos) * 1000, worst(errors.ang) * 180 / .pi,
                          worst(errors.gapPos) * 1000, worst(errors.gapAng) * 180 / .pi, errors.pos.count, errors.gapPos.count))
        out.append(String(format: "paddle small tags: learned turns %@ (true 68:1 69:3), worst %.1f mm / %.2f° from them (%d poses)",
                          fusion.smallTurns.sorted { $0.key < $1.key }.map { "\($0.key):\($0.value)" }.joined(separator: " "),
                          worst(smallErrors.pos) * 1000, worst(smallErrors.ang) * 180 / .pi, smallErrors.pos.count))
        return out
    }
}

extension PingPong {
    /// Self-test: the table at the world origin.
    func placeForTest() { place(head: SIMD3<Float>(0, Float(settings.tableDrop), Float(0.45 + Table.halfLength)), yaw: 0) }
}
