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
        var records: [Int] = []
        game.onRecord = { records.append($0) }
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
            // The posted score: my shot landed on the opponent's half once, so the record is 1 (posted
            // once), and the run starts again from 0 when the rally ends.
            let scoreOK = records == [1] && game.record == 1 && game.streak == 0
            out.append("\(scoreOK ? "" : "FAIL ")continuous hits: record \(game.record) posted as \(records) (expect [1]), streak after the rally \(game.streak) (expect 0)")
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
        out.append(tableAnchorTest())
        out.append(stickmanTest())
        out.append(mqttPacketTest())
        return out
    }

    /// 7. The table on its tag: noisy sightings of a still tag (2 cm, 3°, from 1-2 m) keep it steady;
    ///    the tag moved 40 cm moves it (after a few sightings, gliding).
    /// The stickman through one of the opponent's returns: its hand stays on the paddle's grip, its feet
    /// on the floor, its body behind the end line, and its bones keep their lengths.
    /// The MQTT packets against bytes worked out by hand from the 3.1.1 spec.
    static func mqttPacketTest() -> String {
        let connect = MQTTPacket.connect(clientID: "a", keepAlive: 30)
        let expectConnect: [UInt8] = [0x10, 0x0D, 0x00, 0x04, 0x4D, 0x51, 0x54, 0x54, 0x04, 0x02, 0x00, 0x1E, 0x00, 0x01, 0x61]
        let publish = MQTTPacket.publish(topic: "a/b", payload: "7.0", retain: true)
        let expectPublish: [UInt8] = [0x31, 0x08, 0x00, 0x03, 0x61, 0x2F, 0x62, 0x37, 0x2E, 0x30]
        let ok = connect == expectConnect && publish == expectPublish
            && MQTTPacket.remainingLength(321) == [0xC1, 0x02] && MQTTPacket.remainingLength(127) == [0x7F]
            && MQTTPacket.connectAck([0x20, 0x02, 0x00, 0x00]) == 0
        return ok ? "mqtt packets: connect, publish (retained), remaining length and connack match the spec"
                  : "FAIL mqtt packets: \(connect) \(publish)"
    }

    static func stickmanTest() -> String {
        // A player's shot (Opponent.aim, mirrored to fly toward the opponent), flown to the net.
        guard let (v, w) = Opponent.aim(from: V3(0.1, 0.3, -1.3), to: V3(-0.2, Ball.radius, 0.7), speed: 6, topspin: 80)
        else { return "FAIL stickman: no shot" }
        var ball = Ball(at: V3(0.1, 0.3, 1.3))
        ball.v = V3(v.x, v.y, -v.z)
        ball.w = V3(-w.x, -w.y, w.z)
        var t = 0.0
        while ball.p.z > 0 { Physics.fly(&ball); t += Physics.dt }
        let opponent = Opponent()
        opponent.misses = false
        opponent.plan(ball: ball, time: t)
        opponent.waitForPlan()
        var man = Stickman()
        var handGap = 0.0, footLow = 0.0, front = -9.0, boneError = 0.0, reachGap = 0.0
        var moved = 0.0, first: V3?
        for _ in 0..<150 {
            t += 1.0 / 60
            opponent.update(time: t)
            man.update(paddle: opponent.paddle, time: t)
            let j = man.joints, p = opponent.paddle
            let grip = p.center - p.basis.y * (PaddleShape.semiLength + PaddleShape.handleLength * 0.6)
            handGap = max(handGap, vlength(j[5] - grip))
            reachGap = max(reachGap, vlength(grip - j[3]) - Stickman.upperArm - Stickman.forearm)
            footLow = max(footLow, abs(min(j[10].y, j[12].y) - Table.floorY - 0.02))
            front = max(front, j[2].z)
            for (a, b, l) in [(3, 4, Stickman.upperArm), (4, 5, Stickman.forearm), (2, 9, Stickman.thigh), (9, 10, Stickman.shin)] {
                boneError = max(boneError, abs(vlength(j[a] - j[b]) - l))
            }
            if first == nil { first = j[2] }
            moved = max(moved, vlength(j[2] - first!))
        }
        let ok = handGap < 0.03 && boneError < 0.002 && front < -Table.halfLength
        return String(format: "%@stickman: hand %.1f cm from the grip at worst, bones off %.1f mm, lowest foot %.1f cm off the floor, pelvis z <= %.2f (end line %.2f), moved %.2f m",
                      ok ? "" : "FAIL ", handGap * 100, boneError * 1000, footLow * 100, front, -Table.halfLength, moved)
    }

    static func tableAnchorTest() -> String {
        var rng = GameRandom(seed: 5)
        var anchor = TableAnchor()
        let camera = V3(0, 1.6, 1.5), head = camera
        func sighting(at centre: V3) -> [(rotation: Quat, position: V3)] {
            // Lying flat (z down), one edge along -z, with noise; the planar ambiguity's twin tilted.
            let flat = Quat(rotationVector: V3(.pi / 2, 0, 0)) * Quat(rotationVector: V3(0, 0, rng.uniform(-0.05, 0.05)))
            let tilt = Quat(rotationVector: V3(rng.uniform(-0.03, 0.03), 0, rng.uniform(-0.03, 0.03)))
            let noise = V3(rng.uniform(-0.02, 0.02), rng.uniform(-0.01, 0.01), rng.uniform(-0.02, 0.02))
            return [((tilt * flat).normalized, centre + noise), ((Quat(rotationVector: V3(0.5, 0, 0)) * flat).normalized, centre)]
        }
        var t = 0.0, steady: [TableAnchor.Pose] = [], movedAt: Double?, placedAt: Double?
        let first = V3(0, 0.76, 0), second = V3(0.4, 0.76, 0)
        while t < 9 {
            t += 1.0 / 30
            anchor.add(candidates: sighting(at: t < 5 ? first : second), camera: camera, head: head, time: t)
            if let s = anchor.shown(at: t) {
                if placedAt == nil { placedAt = t }
                if t > 1.5, t < 5 { steady.append(s) }
                if t > 5, movedAt == nil, vlength(s.centre - second) < 0.02 { movedAt = t }
            }
        }
        // Held: the furthest from the true spot, the heading's error, and how far the far end of the table
        // (2 m away) travelled in total while the tag lay still (the wobble that makes one dizzy).
        let worst = steady.map { vlength($0.centre - first) }.max() ?? .nan
        let yawError = steady.map { abs(TableAnchor.wrap($0.yaw)) }.max() ?? .nan
        var farTravel = 0.0
        for i in 1..<max(steady.count, 1) {
            func far(_ p: TableAnchor.Pose) -> V3 { p.centre + V3(-sin(p.yaw), 0, -cos(p.yaw)) * 2 }
            farTravel += vlength(far(steady[i]) - far(steady[i - 1]))
        }
        let ok = worst < 0.02 && yawError < 2 * Double.pi / 180 && farTravel < 0.05 && movedAt != nil
        return String(format: "%@table anchor: placed after %.2f s; still tag held within %.1f mm, %.2f° (noise 20 mm, 3°), far end moved %.1f mm in 3.5 s; moved tag followed after %.2f s",
                      ok ? "" : "FAIL ", placedAt ?? .nan, worst * 1000, yawError * 180 / .pi, farTravel * 1000,
                      (movedAt ?? .nan) - 5)
    }

    /// 6. The smart paddle: a simulated paddle turned in view for the calibration (unknown IMU mount,
    ///    tag turn and clock offset, a drifting IMU heading, noisy and sometimes ambiguous tag views),
    ///    then played with, with the tag out of view for 0.4 s; then turned over so only the two small
    ///    forehand tags are seen (their turns unknown at first).
    static func paddleFusion() -> [String] {
        var rng = GameRandom(seed: 11)
        func noise(_ scale: Double) -> V3 { V3(rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)) * scale }
        var accelRng = GameRandom(seed: 23)
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
                // The accelerometer at the blade centre: its acceleration plus 1 g up, in the IMU's frame,
                // with noise and a bias.
                func centre(_ t: Double) -> V3 { let (r, g) = paddle(t); return g - r.act(PaddleFusion.gripFromCentre) }
                let h = 1e-3
                let a = (centre(nextIMU + h) - centre(nextIMU) * 2 + centre(nextIMU - h)) / (h * h)
                let trueIMU = heading.conjugate * (paddle(nextIMU).rotation * tagFromPaddle.conjugate) * mount.conjugate
                let f = trueIMU.conjugate.act(heading.conjugate.act(a + V3(0, 9.81, 0))) / 9.81
                    + V3(0.02, -0.01, 0.015) + V3(accelRng.uniform(-1, 1), accelRng.uniform(-1, 1), accelRng.uniform(-1, 1)) * 0.02
                fusion.add([PaddleIMUSample(time: nextIMU + timeOffset, orientation: b, rate: rate, still: false, accel: f)])
                nextIMU += 0.005
            }
            fusion.tick(time: t)
            // A wrong big-tag turn (as an unlucky calibration might find): the small tags must fix it.
            if t >= 13, t - dt < 13, var wrong = fusion.calibration {
                wrong.tagTurns = (wrong.tagTurns + 2) % 4
                fusion.calibration = wrong
            }
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
                else if t < 13 { errors.pos.append(pos); errors.ang.append(ang) }
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
        out.append("paddle big tag turns after a wrong one at 13 s: \(fusion.calibration?.tagTurns ?? -1) (true \(turns))")
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
