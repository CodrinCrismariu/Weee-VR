import Foundation

// The smart paddle: its IMU (orientation at 200 Hz) fused with its AprilTag (seen by the camera) and
// the right hand, into the paddle the game plays with. Standard library only, like the rest of the
// game core, so the self-test runs it on Linux.

/// Unit quaternion (w, x, y, z).
struct Quat: Equatable {
    var w: Double, x: Double, y: Double, z: Double

    static let identity = Quat(w: 1, x: 0, y: 0, z: 0)

    init(w: Double, x: Double, y: Double, z: Double) {
        self.w = w; self.x = x; self.y = y; self.z = z
    }

    init(w: Double, v: V3) { self.init(w: w, x: v.x, y: v.y, z: v.z) }

    /// Rotation by |r| radians about r.
    init(rotationVector r: V3) {
        let angle = vlength(r)
        if angle < 1e-12 {
            self = Quat(w: 1, v: r / 2).normalized
        } else {
            self.init(w: cos(angle / 2), v: r * (sin(angle / 2) / angle))
        }
    }

    /// From a rotation's columns (the frame's axes in the parent).
    init(basis b: Basis) {
        let m00 = b.x.x, m10 = b.x.y, m20 = b.x.z
        let m01 = b.y.x, m11 = b.y.y, m21 = b.y.z
        let m02 = b.z.x, m12 = b.z.y, m22 = b.z.z
        let trace = m00 + m11 + m22
        if trace > 0 {
            let s = (trace + 1).squareRoot() * 2
            self.init(w: s / 4, x: (m21 - m12) / s, y: (m02 - m20) / s, z: (m10 - m01) / s)
        } else if m00 > m11, m00 > m22 {
            let s = (1 + m00 - m11 - m22).squareRoot() * 2
            self.init(w: (m21 - m12) / s, x: s / 4, y: (m01 + m10) / s, z: (m02 + m20) / s)
        } else if m11 > m22 {
            let s = (1 + m11 - m00 - m22).squareRoot() * 2
            self.init(w: (m02 - m20) / s, x: (m01 + m10) / s, y: s / 4, z: (m12 + m21) / s)
        } else {
            let s = (1 + m22 - m00 - m11).squareRoot() * 2
            self.init(w: (m10 - m01) / s, x: (m02 + m20) / s, y: (m12 + m21) / s, z: s / 4)
        }
        self = normalized
    }

    var v: V3 { V3(x, y, z) }
    var conjugate: Quat { Quat(w: w, x: -x, y: -y, z: -z) }
    var normalized: Quat {
        let n = (w * w + x * x + y * y + z * z).squareRoot()
        return n > 1e-12 ? Quat(w: w / n, x: x / n, y: y / n, z: z / n) : .identity
    }
    func dot(_ o: Quat) -> Double { w * o.w + x * o.x + y * o.y + z * o.z }
    static prefix func - (q: Quat) -> Quat { Quat(w: -q.w, x: -q.x, y: -q.y, z: -q.z) }

    static func * (a: Quat, b: Quat) -> Quat {
        Quat(w: a.w * b.w - vdot(a.v, b.v), v: b.v * a.w + a.v * b.w + vcross(a.v, b.v))
    }

    /// `p` rotated.
    func act(_ p: V3) -> V3 {
        let t = vcross(v, p) * 2
        return p + t * w + vcross(v, t)
    }

    var basis: Basis { Basis(x: act(V3(1, 0, 0)), y: act(V3(0, 1, 0)), z: act(V3(0, 0, 1))) }

    var rotationVector: V3 {
        let q = w < 0 ? -self : self
        let s = vlength(q.v)
        if s < 1e-12 { return q.v * 2 }
        return q.v / s * (2 * atan2(s, q.w))
    }

    /// Rotation angle, radians (0...pi).
    var angle: Double { 2 * atan2(vlength(v), abs(w)) }

    func slerp(_ to: Quat, _ t: Double) -> Quat {
        let b = dot(to) < 0 ? -to : to
        return (self * Quat(rotationVector: (conjugate * b).rotationVector * t)).normalized
    }

    static func angle(_ a: Quat, _ b: Quat) -> Double { (a.conjugate * b).angle }
}

/// Eigen-decomposition of a small symmetric matrix by Jacobi rotations: (values, vectors as columns).
func symmetricEigen(_ m: [[Double]]) -> (values: [Double], vectors: [[Double]]) {
    let n = m.count
    var a = m
    var v = (0..<n).map { i in (0..<n).map { $0 == i ? 1.0 : 0.0 } }
    for _ in 0..<50 {
        var off = 0.0
        for p in 0..<n { for q in 0..<n where q != p { off += a[p][q] * a[p][q] } }
        if off < 1e-22 { break }
        for p in 0..<(n - 1) {
            for q in (p + 1)..<n where abs(a[p][q]) > 1e-300 {
                let theta = (a[q][q] - a[p][p]) / (2 * a[p][q])
                let t = (theta >= 0 ? 1.0 : -1.0) / (abs(theta) + (theta * theta + 1).squareRoot())
                let c = 1 / (t * t + 1).squareRoot(), s = t * c
                for k in 0..<n {
                    let akp = a[k][p], akq = a[k][q]
                    a[k][p] = c * akp - s * akq
                    a[k][q] = s * akp + c * akq
                }
                for k in 0..<n {
                    let apk = a[p][k], aqk = a[q][k]
                    a[p][k] = c * apk - s * aqk
                    a[q][k] = s * apk + c * aqk
                }
                for k in 0..<n {
                    let vkp = v[k][p], vkq = v[k][q]
                    v[k][p] = c * vkp - s * vkq
                    v[k][q] = s * vkp + c * vkq
                }
            }
        }
    }
    return ((0..<n).map { a[$0][$0] }, v)
}

/// The rotation R that best takes each `a` to its `b` (least squares, Horn's quaternion method).
func fitRotation(_ pairs: [(a: V3, b: V3)]) -> Quat {
    var s = [[Double]](repeating: [0, 0, 0], count: 3)        // s[i][j] = sum a_i b_j
    for (a, b) in pairs {
        let av = [a.x, a.y, a.z], bv = [b.x, b.y, b.z]
        for i in 0..<3 { for j in 0..<3 { s[i][j] += av[i] * bv[j] } }
    }
    let (xx, xy, xz) = (s[0][0], s[0][1], s[0][2])
    let (yx, yy, yz) = (s[1][0], s[1][1], s[1][2])
    let (zx, zy, zz) = (s[2][0], s[2][1], s[2][2])
    let n: [[Double]] = [
        [xx + yy + zz, yz - zy, zx - xz, xy - yx],
        [yz - zy, xx - yy - zz, xy + yx, zx + xz],
        [zx - xz, xy + yx, -xx + yy - zz, yz + zy],
        [xy - yx, zx + xz, yz + zy, -xx - yy + zz],
    ]
    let (values, vectors) = symmetricEigen(n)
    let best = values.indices.max { values[$0] < values[$1] }!
    return Quat(w: vectors[0][best], x: vectors[1][best], y: vectors[2][best], z: vectors[3][best]).normalized
}

// MARK: - Inputs and the saved calibration

/// One smart-paddle IMU sample.
struct PaddleIMUSample {
    var time: Double            // host clock, as the link maps the MCU's clock (before `timeOffset`)
    var orientation: Quat       // IMU earth <- IMU (earth z up; heading arbitrary and slowly drifting)
    var rate: V3                // rad/s, IMU frame
    var still: Bool
}

/// The tag seen in one camera frame: the pose candidates (the planar ambiguity), best fit first.
struct PaddleTagSighting {
    struct Candidate {
        var rotation: Quat      // world <- tag (tag: x right, y down, z into the tag, AprilTag's frame)
        var position: V3        // tag centre, world
        var error: Double       // AprilTag's object-space error
    }
    var id = 0                  // the tag's ID (`PaddleFusion.bigTagIDs`, `smallTags`)
    var time: Double            // capture time (host clock)
    var candidates: [Candidate]
}

/// Measured once per paddle build (Settings > Smart paddle > Calibrate paddle) and saved.
struct PaddleCalibration: Codable, Equatable {
    var mount: [Double]         // w x y z: the tag's frame in the IMU's frame (IMU <- tag)
    var timeOffset: Double      // s, IMU clock minus camera clock (the link's stamps run this much late)
    var tagTurns: Int           // the tag's in-plane rotation on the blade, quarter turns
    var residualDeg: Double     // fit quality when it was measured

    var mountQuat: Quat { Quat(w: mount[0], x: mount[1], y: mount[2], z: mount[3]).normalized }
}

// MARK: - Fusion

/// The paddle from its IMU, tag and the right hand.
///
/// - Orientation: `world <- paddle = heading * imu(t + timeOffset) * mount * turns`. The IMU gives it at
///   200 Hz (fused at 1 kHz on the paddle); `heading` (IMU earth -> world) is re-estimated whenever the
///   tag is seen, which removes the IMU's heading drift; `mount` and `timeOffset` are the saved
///   calibration.
/// - Position: the blade swings mostly by rotating about the hand, so the paddle is placed from its
///   grip point (the middle of the handle) plus the IMU's rotation. The grip point comes from a
///   constant-velocity Kalman filter fed by the tag (exact, when seen) and the right palm (a few cm,
///   with the palm -> grip offset learned while the tag is seen).
/// - Calibration (`beginCalibration`): while the paddle is turned in view, pairs of tag rotations and
///   IMU rotations give the mount (relative rotations: tag_i^T tag_j = mount^T imu_i^T imu_j mount, the
///   hand-eye equation, solved by Horn's method on their axes) and the clock offset (the one with the
///   smallest residual); the palm -> tag direction gives which way the tag is turned on the blade.
final class PaddleFusion {
    /// Paddle frame (`PaddleShape`): from the tag centre to the blade centre. The tag is 20 mm toward
    /// the handle from the blade centre, on the backhand face (the tag's z, into it, is the forehand
    /// normal).
    static let tagToCentre = V3(0, 0.020, 0.006)
    /// Paddle frame: the grip point (middle of the handle) from the blade centre.
    static let gripFromCentre = V3(0, -PaddleShape.gripReach, 0)
    /// The big tag (backhand face).
    static let bigTagIDs: Set<Int> = [0]
    /// The small tags on the forehand face, from the blade centre (paddle frame): 68 on the right and
    /// 69 on the left as the forehand face looks at you, 55 mm out, on the rubber.
    static let smallTags: [Int: V3] = [68: V3(0.055, 0, PaddleShape.halfThickness),
                                       69: V3(-0.055, 0, PaddleShape.halfThickness)]
    /// How a tag can lie on the forehand face (tag <- paddle): turned over (its z, into it, is the
    /// paddle's -z), then quarter turns in its plane.
    static let forehandPlacements: [Quat] = (0..<4).map {
        (Quat(rotationVector: V3(.pi, 0, 0)) * Quat(rotationVector: V3(0, 0, Double($0) * .pi / 2))).normalized
    }

    struct Progress {
        var sightings: Int
        var seconds: Double
        var message: String
    }

    var calibration: PaddleCalibration? {
        didSet { if calibration?.mount != oldValue?.mount { heading = nil } }
    }
    /// Settings override of the tag's quarter turns (nil: the calibration's).
    var tagTurnsOverride: Int?
    /// With no saved calibration, learn it in the background whenever the tag is seen.
    var autoCalibrate = true
    var onCalibrated: ((PaddleCalibration) -> Void)?
    /// Learned quarter turns of the small tags (ID -> 0...3, `forehandPlacements`), saved.
    var smallTurns: [Int: Int] = [:]
    var onSmallTagsLearned: (([Int: Int]) -> Void)?
    var onCalibrationFailed: ((String) -> Void)?
    var onEvent: ((String) -> Void)?

    private(set) var progress: Progress?
    private(set) var source = "none"

    private var imu: [PaddleIMUSample] = []
    private var heading: Quat?                  // IMU earth -> world
    private var headingTime = 0.0
    private var headingOutliers = 0
    private var grip = GripFilter()
    private var palmToGrip = V3(0, 0, -0.012)   // paddle frame (the hand grip's blade offset)
    private var lastTag: (time: Double, rotation: Quat, position: V3)?
    private var lastPalm: (time: Double, position: V3)?
    private var rawPaddle: (time: Double, rotation: Quat, centre: V3)?       // from the latest tag alone
    private var previousRawPaddle: (time: Double, rotation: Quat, centre: V3)?
    private var smallSeen: [(id: Int, time: Double, position: V3)] = []
    private var smallVotes: [Int: [Int]] = [:]

    private var run: CalibrationRun?
    private let calibrationTimeout = 20.0

    private var turns: Int { ((tagTurnsOverride ?? calibration?.tagTurns ?? 0) % 4 + 4) % 4 }
    /// paddle frame in the tag frame (tag <- paddle): a quarter-turn about the shared z.
    private var tagFromPaddle: Quat { Quat(rotationVector: V3(0, 0, Double(turns) * .pi / 2)) }

    func reset() {
        imu.removeAll()
        heading = nil
        grip = GripFilter()
        lastTag = nil
        lastPalm = nil
    }

    // MARK: Inputs

    func add(_ samples: [PaddleIMUSample]) {
        for s in samples where imu.last.map({ s.time > $0.time }) ?? true { imu.append(s) }
        if let last = imu.last?.time, let drop = imu.firstIndex(where: { $0.time > last - 2.0 }), drop > 0 {
            imu.removeFirst(drop)
        }
    }

    /// The right palm (world), from the hand tracker, at its capture time. It places the grip only
    /// while the tag isn't seen (the tag is far more precise).
    func addPalm(_ position: V3, time: Double) {
        lastPalm = (time, position)
        if run != nil { return }
        if let tag = lastTag, time - tag.time < 0.1 { return }
        guard let cal = calibration, let q = orientation(at: time, calibration: cal) else { return }
        grip.accelNoise = swingNoise
        grip.update(position + q.act(palmToGrip), sigma: 0.025, time: time)
    }

    /// How freely the grip may move between measurements: little while the paddle turns slowly, a lot
    /// in a swing (the arm swings with the blade), judged by the IMU's rate.
    private var swingNoise: Double {
        let spin = min(vlength(imu.last?.rate ?? .zero) / 10, 1)
        return 2 + 60 * spin * spin
    }

    func add(_ sighting: PaddleTagSighting) {
        guard !sighting.candidates.isEmpty else { return }
        if let offset = Self.smallTags[sighting.id] {
            addSmall(sighting, offset: offset)
            return
        }
        guard Self.bigTagIDs.contains(sighting.id) else { return }
        // The raw paddle, for before calibration: the candidate nearest the last one (the planar
        // ambiguity would otherwise flip it between frames).
        var raw = sighting.candidates[0]
        if let last = rawPaddle, sighting.candidates.count > 1, sighting.time - last.time < 0.3 {
            raw = sighting.candidates.min {
                Quat.angle($0.rotation * tagFromPaddle, last.rotation) < Quat.angle($1.rotation * tagFromPaddle, last.rotation)
            }!
        }
        let rawRotation = (raw.rotation * tagFromPaddle).normalized
        setRawPaddle(time: sighting.time, rotation: rawRotation, centre: raw.position + rawRotation.act(Self.tagToCentre))
        if run != nil {
            collect(sighting)
            return
        }
        guard let cal = calibration else { return }
        let t = sighting.time
        guard let b = imuOrientation(at: t + cal.timeOffset) else { return }

        // The planar ambiguity: the candidate the IMU agrees with (once the heading is known).
        var tag = sighting.candidates[0]
        if let heading, sighting.candidates.count > 1 {
            let predicted = heading * b * cal.mountQuat
            tag = sighting.candidates.min { Quat.angle($0.rotation, predicted) < Quat.angle($1.rotation, predicted) }!
        }
        let paddle = (tag.rotation * tagFromPaddle).normalized
        let centre = tag.position + paddle.act(Self.tagToCentre)
        fuse(paddle: paddle, centre: centre, time: t, imu: b, calibration: cal)
    }

    /// A small tag on the forehand face. Its offset from the blade centre is known; how it is turned
    /// is learned (the first time both small tags are in view, the line from 69 to 68 is the paddle's
    /// x; or from the calibrated IMU), then kept.
    private func addSmall(_ sighting: PaddleTagSighting, offset: V3) {
        let t = sighting.time
        let cal = calibration
        let b = cal.flatMap { imuOrientation(at: t + $0.timeOffset) }
        let imuPaddle: Quat? = (heading != nil && b != nil && cal != nil)
            ? heading! * b! * cal!.mountQuat * tagFromPaddle : nil

        guard let turns = smallTurns[sighting.id] else {
            learnSmall(sighting, imuPaddle: imuPaddle)
            return
        }
        let placement = Self.forehandPlacements[turns]
        // The planar ambiguity: the candidate the IMU agrees with, else the one nearest the last paddle.
        var tag = sighting.candidates[0]
        if sighting.candidates.count > 1, let reference = imuPaddle ?? rawPaddle.flatMap({ t - $0.time < 0.3 ? $0.rotation : nil }) {
            tag = sighting.candidates.min {
                Quat.angle($0.rotation * placement, reference) < Quat.angle($1.rotation * placement, reference)
            }!
        }
        let paddle = (tag.rotation * placement).normalized
        let centre = tag.position - paddle.act(offset)
        setRawPaddle(time: t, rotation: paddle, centre: centre)
        if run == nil, let cal, let b { fuse(paddle: paddle, centre: centre, time: t, imu: b, calibration: cal) }
    }

    private func learnSmall(_ sighting: PaddleTagSighting, imuPaddle: Quat?) {
        let t = sighting.time
        var vote: Int?
        if let other = smallSeen.first(where: { $0.id != sighting.id && abs($0.time - t) < 0.002 }) {
            // Both in this frame: the paddle's x points from 69 to 68.
            let a68 = sighting.id == 68 ? sighting.candidates[0].position : other.position
            let a69 = sighting.id == 68 ? other.position : sighting.candidates[0].position
            let x = vnormalize(a68 - a69)
            if abs(vlength(a68 - a69) - 0.11) < 0.03 {
                var best = (score: -2.0, k: 0)
                for c in sighting.candidates {
                    for k in 0..<4 {
                        let paddle = c.rotation * Self.forehandPlacements[k]
                        let score = vdot(paddle.act(V3(1, 0, 0)), x)
                        if score > best.score { best = (score, k) }
                    }
                }
                if best.score > 0.85 { vote = best.k }
            }
        } else if let imuPaddle {
            var best = (angle: Double.infinity, k: 0)
            for c in sighting.candidates {
                for k in 0..<4 {
                    let angle = Quat.angle(c.rotation * Self.forehandPlacements[k], imuPaddle)
                    if angle < best.angle { best = (angle, k) }
                }
            }
            if best.angle < 25 * .pi / 180 { vote = best.k }
        }
        smallSeen.append((sighting.id, t, sighting.candidates[0].position))
        if smallSeen.count > 8 { smallSeen.removeFirst(smallSeen.count - 8) }
        guard let vote else { return }
        smallVotes[sighting.id, default: [0, 0, 0, 0]][vote] += 1
        let votes = smallVotes[sighting.id]!
        let total = votes.reduce(0, +)
        if total >= 10, let k = votes.indices.max(by: { votes[$0] < votes[$1] }), Double(votes[k]) >= 0.7 * Double(total) {
            smallTurns[sighting.id] = k
            onEvent?("paddle: tag \(sighting.id) placement learned (\(k) quarter turns, \(votes))")
            onSmallTagsLearned?(smallTurns)
        }
    }

    /// One paddle pose seen by a tag: keeps the IMU heading and the grip point.
    private func fuse(paddle: Quat, centre: V3, time t: Double, imu b: Quat, calibration cal: PaddleCalibration) {
        // Heading: the IMU's earth frame in the world, smoothed (1/2 s), restarted after a run of
        // disagreeing sightings (the paddle's board restarted, say).
        let observed = (paddle * (cal.mountQuat * tagFromPaddle).conjugate * b.conjugate).normalized
        if let current = heading {
            if Quat.angle(current, observed) > 20 * .pi / 180 {
                headingOutliers += 1
                if headingOutliers >= 6 {
                    heading = observed
                    headingOutliers = 0
                    onEvent?("paddle: heading re-acquired")
                }
            } else {
                headingOutliers = 0
                let alpha = 1 - exp(-min(max(t - headingTime, 0), 0.5) / 0.5)
                heading = current.slerp(observed, alpha)
            }
        } else {
            heading = observed
            onEvent?("paddle: heading acquired")
        }
        headingTime = t

        // Grip point from the tag (its own rotation is exact at this instant).
        let gripPoint = centre + paddle.act(Self.gripFromCentre)
        grip.accelNoise = swingNoise
        grip.update(gripPoint, sigma: 0.004, time: t)
        lastTag = (t, paddle, centre)

        // Learn where the grip is from the palm, for when no tag is seen.
        if let palm = lastPalm, abs(palm.time - t) < 0.02 {
            let offset = paddle.conjugate.act(gripPoint - palm.position)
            if vlength(offset) < 0.12 { palmToGrip += (offset - palmToGrip) * 0.05 }
        }
    }

    private func setRawPaddle(time: Double, rotation: Quat, centre: V3) {
        if let raw = rawPaddle, abs(raw.time - time) < 0.002 { return }      // one per frame
        previousRawPaddle = rawPaddle
        rawPaddle = (time, rotation, centre)
    }

    // MARK: Output

    /// The paddle at `time` (world frame; velocities in real time), or nil when the smart paddle
    /// can't place it (not calibrated, no IMU, or nothing has placed the grip yet).
    func pose(at time: Double) -> PaddleState? {
        // Calibrated: the IMU carries the paddle however long the tags and the hand are gone.
        guard let cal = calibration, let heading, grip.ready,
              let (b, rate) = imuState(at: time + cal.timeOffset) else {
            // Not calibrated yet (it is learning in the background): the paddle follows the tags alone.
            if let tagOnly = tagOnlyPose(at: time) {
                source = "tag only (calibrating)"
                return tagOnly
            }
            source = calibration == nil ? "calibrating: hold the big tag in view and move the paddle"
                : heading == nil ? "waiting for a tag" : "waiting"
            return nil
        }
        let imuToWorld = heading * b
        let paddle = (imuToWorld * cal.mountQuat * tagFromPaddle).normalized
        let (g, gv) = grip.predict(to: time)
        let centre = g - paddle.act(Self.gripFromCentre)
        let omega = imuToWorld.act(rate)
        var state = PaddleState(center: centre, basis: paddle.basis)
        state.velocity = gv + vcross(omega, centre - g)
        state.angular = omega
        state.visible = true
        let tagRecent = lastTag.map { time - $0.time < 0.25 } ?? false
        let handRecent = lastPalm.map { time - $0.time < 0.25 } ?? false
        source = tagRecent ? "imu+tag" : handRecent ? "imu+hand" : "imu (holding position)"
        return state
    }

    /// Where the blade centre is at `time` (world), from the calibrated IMU pose: where to look for tags.
    func predictedCentre(at time: Double) -> V3? { pose(at: time)?.center }

    /// Before calibration: the paddle straight from the latest tag sighting (no IMU), for a moment.
    private func tagOnlyPose(at time: Double) -> PaddleState? {
        guard let raw = rawPaddle, time - raw.time < 0.25 else { return nil }
        var state = PaddleState(center: raw.centre, basis: raw.rotation.basis)
        if let prev = previousRawPaddle, raw.time - prev.time > 1e-3, raw.time - prev.time < 0.1 {
            let dt = raw.time - prev.time
            state.velocity = (raw.centre - prev.centre) / dt
            state.angular = prev.rotation.act((prev.rotation.conjugate * raw.rotation).rotationVector / dt)
        }
        state.visible = true
        return state
    }

    /// Calibrated paddle orientation (world <- paddle) at camera-clock `time`.
    private func orientation(at time: Double, calibration cal: PaddleCalibration) -> Quat? {
        guard let heading, let b = imuOrientation(at: time + cal.timeOffset) else { return nil }
        return heading * b * cal.mountQuat * tagFromPaddle
    }

    // MARK: IMU history

    private func imuOrientation(at t: Double) -> Quat? { imuState(at: t)?.0 }

    /// IMU orientation and rate at IMU-clock `t`: interpolated, or extrapolated with the last rate for
    /// up to 50 ms past the newest sample.
    private func imuState(at t: Double) -> (Quat, V3)? {
        guard let last = imu.last, let first = imu.first, t >= first.time else { return nil }
        if t >= last.time {
            let dt = t - last.time
            guard dt < 0.25 else { return nil }
            let ahead = min(dt, 0.05)
            return ((last.orientation * Quat(rotationVector: last.rate * ahead)).normalized, last.rate)
        }
        var lo = 0, hi = imu.count - 1
        while hi - lo > 1 {
            let mid = (lo + hi) / 2
            if imu[mid].time <= t { lo = mid } else { hi = mid }
        }
        let a = imu[lo], b = imu[hi]
        let f = (t - a.time) / max(b.time - a.time, 1e-6)
        return (a.orientation.slerp(b.orientation, f), a.rate + (b.rate - a.rate) * f)
    }

    // MARK: Calibration

    private struct CalibrationRun {
        var automatic = false               // started on its own (no saved calibration): no time limit
        var requested: Double?
        var sightings: [(time: Double, candidates: [Quat], position: V3)] = []
        var votes = [0, 0, 0, 0]
        var lastTry = 0.0
    }

    func beginCalibration(automatic: Bool = false) {
        run = CalibrationRun(automatic: automatic)
        progress = Progress(sightings: 0, seconds: 0, message: "turn the paddle slowly, tag toward you")
        onEvent?("paddle: calibration started")
    }

    func cancelCalibration() {
        run = nil
        progress = nil
    }

    private func collect(_ sighting: PaddleTagSighting) {
        guard var r = run else { return }
        let t = sighting.time
        // Both pose candidates are kept: a flat tag seen nearly face-on often fits two poses about
        // equally well. The solve picks, per pair of views, the one whose turn matches the gyro's.
        let best = sighting.candidates[0]
        let candidates = sighting.candidates.filter { $0.error <= max(best.error * 20, 1e-6) }.prefix(2).map(\.rotation)
        r.sightings.append((t, Array(candidates), best.position))
        if r.sightings.count > 400 { r.sightings.removeFirst(100) }     // an automatic run keeps the latest views
        // Which way the tag is turned: the paddle's y (toward the tip) points from the palm to the tag.
        if let palm = lastPalm, abs(palm.time - t) < 0.05 {
            let d = vnormalize(best.position - palm.position)
            let ys = (0..<4).map { k in (best.rotation * Quat(rotationVector: V3(0, 0, Double(k) * .pi / 2))).act(V3(0, 1, 0)) }
            let scores = ys.map { vdot($0, d) }
            if let k = scores.indices.max(by: { scores[$0] < scores[$1] }), scores[k] > 0.5 { r.votes[k] += 1 }
        }
        progress?.sightings = r.sightings.count
        run = r
        if r.sightings.count >= 40, t - r.lastTry > 1.0 {
            run?.lastTry = t
            if let result = solve(r) {
                run = nil
                progress = nil
                calibration = result
                onEvent?(String(format: "paddle: calibrated, residual %.2f°, clock offset %.1f ms, tag turns %d (%d views)",
                                result.residualDeg, result.timeOffset * 1000, result.tagTurns, r.sightings.count))
                onCalibrated?(result)
            }
        }
    }

    /// Render thread, every frame: progress and the time limit (which also applies when the tag is
    /// never seen).
    func tick(time: Double) {
        if run == nil, calibration == nil, autoCalibrate { beginCalibration(automatic: true) }
        guard var r = run else { return }
        if r.requested == nil { r.requested = time }
        let elapsed = time - r.requested!
        progress?.seconds = elapsed
        run = r
        guard !r.automatic, elapsed > calibrationTimeout else { return }
        let why = r.sightings.isEmpty ? "the tag was never seen (light, distance, tag toward the camera?)"
            : (progress?.message ?? "not enough turning") + " (\(r.sightings.count) views)"
        run = nil
        progress = nil
        onEvent?("paddle: calibration gave up: \(why)")
        onCalibrationFailed?(why)
    }

    /// 0...1 while calibrating (views collected), nil otherwise.
    var calibrationProgress: Double? {
        guard let p = progress else { return nil }
        return min(Double(p.sightings) / 60, 0.95)
    }

    /// The mount and clock offset that best explain the tag's rotations by the IMU's.
    private func solve(_ r: CalibrationRun) -> PaddleCalibration? {
        let s = r.sightings
        // Pairs of views a moderate turn apart, with the relative rotation (tag frame) for each
        // choice of candidates.
        var pairs: [(i: Int, j: Int, options: [V3])] = []
        for i in s.indices {
            for gap in [2, 4, 7, 11, 16, 24] where i + gap < s.count {
                let j = i + gap
                guard s[j].time - s[i].time < 1.5 else { continue }
                var options: [V3] = []
                for a in s[i].candidates { for b in s[j].candidates { options.append((a.conjugate * b).rotationVector) } }
                if options.contains(where: { vlength($0) > 0.17 && vlength($0) < 2.1 }) { pairs.append((i, j, options)) }
            }
        }
        guard pairs.count >= 30 else {
            progress?.message = "turn the paddle more"
            return nil
        }

        func fit(offset: Double) -> (mount: Quat, residual: Double, data: [(a: V3, b: V3)])? {
            var rows: [(options: [V3], b: V3)] = []
            rows.reserveCapacity(pairs.count)
            for p in pairs {
                guard let bi = imuOrientation(at: s[p.i].time + offset),
                      let bj = imuOrientation(at: s[p.j].time + offset) else { continue }
                let b = (bi.conjugate * bj).rotationVector
                let nb = vlength(b)
                if nb > 0.17, nb < 2.1 { rows.append((p.options, b)) }
            }
            guard rows.count >= 30 else { return nil }
            // First guess: the candidates whose turn angle matches the gyro's (the angle doesn't
            // depend on the mount). Then, a few times, the candidates the fitted mount agrees with,
            // without the pairs well off the fit (a misread tag), and fit again.
            var data = rows.compactMap { row -> (a: V3, b: V3)? in
                let nb = vlength(row.b)
                let a = row.options.min { abs(vlength($0) - nb) < abs(vlength($1) - nb) }!
                return abs(vlength(a) - nb) < 0.07 ? (a, row.b) : nil
            }
            guard data.count >= 30 else { return nil }
            var mount = fitRotation(data)
            var residual = 0.0
            for _ in 0..<3 {
                let chosen = rows.map { row -> (a: V3, b: V3, error: Double) in
                    let a = row.options.min { vlength(mount.act($0) - row.b) < vlength(mount.act($1) - row.b) }!
                    return (a, row.b, vlength(mount.act(a) - row.b))
                }
                let median = chosen.map(\.error).sorted()[chosen.count / 2]
                data = chosen.filter { $0.error <= max(3 * median, 0.03) }.map { ($0.a, $0.b) }
                guard data.count >= 25 else { return nil }
                mount = fitRotation(data)
                let errors = data.map { vlength(mount.act($0.a) - $0.b) }
                residual = errors.reduce(0, +) / Double(errors.count)
            }
            return (mount, residual, data)
        }

        var best: (offset: Double, mount: Quat, residual: Double, data: [(a: V3, b: V3)])?
        var offset = -0.060
        while offset <= 0.040 + 1e-9 {
            if let f = fit(offset: offset), f.residual < best?.residual ?? .infinity { best = (offset, f.mount, f.residual, f.data) }
            offset += 0.002
        }
        guard let coarse = best else {
            progress?.message = "the tag's turns don't match the IMU's yet: keep turning slowly"
            return nil
        }
        offset = coarse.offset - 0.002
        while offset <= coarse.offset + 0.002 + 1e-9 {
            if let f = fit(offset: offset), f.residual < best!.residual { best = (offset, f.mount, f.residual, f.data) }
            offset += 0.0005
        }
        let result = best!
        // Turned about at least two independent axes.
        var spread = [[Double]](repeating: [0, 0, 0], count: 3)
        for d in result.data {
            let u = vnormalize(d.b), uv = [u.x, u.y, u.z]
            for i in 0..<3 { for j in 0..<3 { spread[i][j] += uv[i] * uv[j] / Double(result.data.count) } }
        }
        guard symmetricEigen(spread).values.sorted()[1] > 0.08 else {
            progress?.message = "turn the paddle about another axis too"
            return nil
        }
        // The MPU6050's gyro scale is off by a few %, so the fit can't be much tighter than this.
        let residualDeg = result.residual * 180 / .pi
        guard residualDeg < 4.0 else {
            progress?.message = String(format: "fit %.1f° (needs < 4°): keep turning slowly", residualDeg)
            return nil
        }
        // b = mount a: mount takes tag-frame axes to IMU-frame axes (IMU <- tag).
        let m = result.mount
        let turnsVoted = r.votes.indices.max { r.votes[$0] < r.votes[$1] }!
        let turns = r.votes[turnsVoted] >= 5 ? turnsVoted : (calibration?.tagTurns ?? 0)
        return PaddleCalibration(mount: [m.w, m.x, m.y, m.z], timeOffset: result.offset, tagTurns: turns,
                                 residualDeg: residualDeg)
    }

    var summary: String {
        var parts: [String] = []
        if let cal = calibration {
            parts.append(String(format: "mount %.1f° dt %+.0fms turns %d", cal.residualDeg, cal.timeOffset * 1000, turns))
        } else {
            parts.append("not calibrated")
        }
        if let p = progress { parts.append(String(format: "calibrating %.0fs %d views: %@", p.seconds, p.sightings, p.message)) }
        parts.append("small tags " + Self.smallTags.keys.sorted().map { id in smallTurns[id].map { "\(id):\($0)" } ?? "\(id):?" }.joined(separator: " "))
        parts.append(source)
        return parts.joined(separator: " · ")
    }
}

/// Constant-velocity Kalman filter for a point (the same variances on all three axes).
struct GripFilter {
    private(set) var position = V3.zero
    private(set) var velocity = V3.zero
    private(set) var time = 0.0
    private(set) var ready = false
    // Covariance [[pp, pv], [pv, vv]] per axis.
    private var pp = 1.0, pv = 0.0, vv = 1.0
    /// Acceleration noise spectral density, m^2/s^3: how freely the hand moves between measurements.
    var accelNoise = 2.0

    mutating func update(_ z: V3, sigma: Double, time t: Double) {
        guard ready else {
            position = z
            velocity = .zero
            time = t
            pp = sigma * sigma; pv = 0; vv = 1
            ready = true
            return
        }
        let dt = t - time
        if dt > 0 {
            position += velocity * dt
            let q = accelNoise
            let npp = pp + 2 * dt * pv + dt * dt * vv + q * dt * dt * dt / 3
            let npv = pv + dt * vv + q * dt * dt / 2
            let nvv = vv + q * dt
            pp = npp; pv = npv; vv = nvv
            time = t
        }
        // A measurement older than the filter (another camera frame's) is used as if it were now.
        let s = pp + sigma * sigma
        let kp = pp / s, kv = pv / s
        let innovation = z - position
        position += innovation * kp
        velocity += innovation * kv
        let npp = (1 - kp) * pp
        let npv = (1 - kp) * pv
        let nvv = vv - kv * pv
        pp = npp; pv = npv; vv = nvv
        if vlength(velocity) > 6 { velocity = vnormalize(velocity) * 6 }
    }

    /// Position and velocity predicted to `t` (at most 0.1 s ahead).
    func predict(to t: Double) -> (V3, V3) {
        let dt = min(max(t - time, 0), 0.1)
        return (position + velocity * dt, velocity)
    }
}
