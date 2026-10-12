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
    var accel: V3? = nil        // g, IMU frame (specific force: +1 g up at rest); nil from old firmware
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
    /// Paddle frame: the point the paddle turns about, from the blade centre: where the hand meets the
    /// handle (a little behind its middle). In a quick turn with no tag in view, the blade swings about
    /// it; measured against a real recording (scripts/paddle-replay), this distance gave the fewest
    /// jumps without drifting from the tags.
    static let gripFromCentre = V3(0, -0.18, 0)
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
        var fraction = 0.0          // the ring: half for views, then turning about two axes, then the fit
    }

    var calibration: PaddleCalibration? {
        didSet { if calibration?.mount != oldValue?.mount { heading = nil } }
    }
    /// Settings override of the tag's quarter turns (nil: the calibration's).
    var tagTurnsOverride: Int?
    /// With no saved calibration, learn it in the background whenever the tag is seen.
    var autoCalibrate = true
    var onCalibrated: ((PaddleCalibration) -> Void)?
    /// A background improvement of the calibration (refined mount, corrected tag turns): saved quietly,
    /// no buzz or ring.
    var onCalibrationImproved: ((PaddleCalibration) -> Void)?
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
    private var grip = PivotTracker()
    /// Jumps of the shown paddle (the tags back after a gap) are eased out over ~0.1 s.
    private var shownOffset = V3.zero
    private var shownOffsetTime: Double?
    private var shownRate = V3.zero
    /// The frame's time (from `tick`): corrections ease out on it, so the paddle drawn on the camera
    /// image and the one the physics samples ahead carry the same offset.
    private var frameClock: Double?
    private var useFrameClock: Bool { FusionParams.current.glideClock == 1 && frameClock != nil }              // the offset's own rate when it started (glideShape 1)
    private var palmToGrip: V3?                     // paddle frame, learned while a tag and the hand are seen
    private var lastFusedTag: Int?
    /// The blade centre from the accelerometer (the IMU is almost at the blade centre), corrected by the
    /// tags and the hand; nil until there's an accelerometer and a first tag.
    private var centre = CentreFilter()
    private var ins = InsFilter()
    private var useIns: Bool { FusionParams.current.filter == 1 }
    /// The player's head (world), for the arm's reach: set every frame by the engine.
    var head: V3?
    private var lastAccelTime = -1.0
    /// Learned corrections of the small tags' placement (paddle frame, added to their nominal offset).
    private(set) var sideCorrection: [Int: V3] = [:]
    private var smoothPalm: (time: Double, position: V3)?
    private var lastTag: (time: Double, rotation: Quat, position: V3)?
    private var lastPalm: (time: Double, position: V3)?
    private var rawPaddle: (time: Double, rotation: Quat, centre: V3)?       // from the latest tag alone
    private var previousRawPaddle: (time: Double, rotation: Quat, centre: V3)?
    private var smallSeen: [(id: Int, time: Double, candidates: [PaddleTagSighting.Candidate])] = []
    private var smallVotes: [Int: [Int]] = [:]

    private var run: CalibrationRun?
    private let calibrationTimeout = 20.0

    private var turns: Int { ((tagTurnsOverride ?? calibration?.tagTurns ?? votedTurns ?? 0) % 4 + 4) % 4 }
    /// While calibrating: which way the big tag is turned, from where the hand is (the paddle's y points
    /// from the palm past the tag), once there are enough votes.
    private var votedTurns: Int? {
        guard let votes = run?.votes, let k = votes.indices.max(by: { votes[$0] < votes[$1] }), votes[k] >= 5 else { return nil }
        return k
    }
    /// The big tag's heading (IMU earth -> world) as last seen: checks the big tag's turns against the
    /// small tags' (the IMU links the two sides).
    private var lastBigHeading: (heading: Quat, time: Double)?
    private var bigTurnVotes = [0, 0, 0, 0]
    /// paddle frame in the tag frame (tag <- paddle): a quarter-turn about the shared z.
    private var tagFromPaddle: Quat { Quat(rotationVector: V3(0, 0, Double(turns) * .pi / 2)) }

    func reset() {
        imu.removeAll()
        heading = nil
        grip = PivotTracker()
        centre = CentreFilter()
        ins = InsFilter()
        shownOffset = .zero
        shownRate = .zero
        shownOffsetTime = 0
        palmToGrip = nil
        lastTag = nil
        lastPalm = nil
    }

    // MARK: Inputs

    func add(_ samples: [PaddleIMUSample]) {
        for s in samples where imu.last.map({ s.time > $0.time }) ?? true {
            imu.append(s)
            propagate(s)
        }
        if let last = imu.last?.time, let drop = imu.firstIndex(where: { $0.time > last - 2.0 }), drop > 0 {
            imu.removeFirst(drop)
        }
    }

    /// The accelerometer moves the blade centre on: its specific force turned into the world (the IMU's
    /// orientation) less gravity. Between tag sightings the velocity slowly leaks away (the
    /// accelerometer's bias would otherwise make it drift), faster when the paddle is held still.
    private func propagate(_ s: PaddleIMUSample) {
        guard let f = s.accel, let cal = calibration, let heading, useIns ? ins.ready : centre.ready else { return }
        let t = s.time - cal.timeOffset
        // Gravity off in the IMU's own earth frame (its z is the IMU's own, precise, down), then into the
        // world: a few degrees of error in the tag-made heading then only turn the motion a little,
        // instead of leaking a share of gravity into it.
        // The IMU sits a little off the blade centre: in a turn it also feels the turn's own
        // acceleration (angular acceleration x r, and the centripetal w x (w x r)); taken off, what's left
        // is the blade centre's.
        let r = Self.imuFromCentre
        var fc = f
        if let p = previousRate, s.time > p.time {
            let alpha = (s.rate - p.rate) / max(s.time - p.time, 1e-3)
            fc = f - (vcross(alpha, r) + vcross(s.rate, vcross(s.rate, r))) / Self.gravity
        }
        previousRate = (s.time, s.rate)
        let world = heading.act((s.orientation.act(fc) - V3(0, 0, 1)) * Self.gravity)
        let sinceTag = lastTag.map { t - $0.time } ?? .infinity
        let k = FusionParams.current
        var damping = sinceTag > k.dampAfter ? k.damping : 0.0
        if s.still { damping = k.stillDamping }
        // Hard swings are when the accelerometer is least trustworthy (scale error, clipping, the lever
        // arm's estimate): its noise grows with the acceleration and the spin.
        let aRel = vlength(fc) - 1, spin = vlength(s.rate)
        let noise = 1 + (k.motionA * aRel) * (k.motionA * aRel) + (k.motionW * spin) * (k.motionW * spin)
        if useIns {
            ins.propagate(InsFilter.Input(time: t, specific: fc, imuToEarth: s.orientation, heading: heading,
                                          damping: damping, noise: noise))
        } else {
            centre.propagate(to: t, accel: world, damping: damping, noise: noise)
        }
        // The arm's reach: with no tag for a while, the blade centre can't be further from the shoulder
        // (just below the head) than the arm and paddle allow; past that it's pulled back in.
        let reach = FusionParams.current.armReach
        if reach > 0, sinceTag > 0.15, let head, sampleCount % 4 == 0,
           let now = useIns ? ins.state(at: t)?.p : centre.state(at: t)?.p {
            let shoulder = head + V3(0, -0.25, 0)
            let d = now - shoulder
            if vlength(d) > reach {
                let z = shoulder + vnormalize(d) * reach
                let sigma = FusionParams.current.armSigma
                glide(-(useIns ? ins.correct(z, R: Mat.identity(3, sigma * sigma), at: t)
                              : centre.correct(z, sigma: sigma, at: t)), at: t)
            }
        }
        sampleCount += 1
        lastAccelTime = t
    }

    private var previousRate: (time: Double, rate: V3)?
    private var sampleCount = 0
    /// Where the IMU sits from the blade centre, in its own frame: ~7 cm along its y (fitted against a
    /// recording with the tags hidden in windows, scripts/paddle-replay).
    static var imuFromCentre: V3 { FusionParams.current.imuFromCentre }
    static let gravity = 9.81

    /// The accelerometer has been heard from lately (new firmware, link up).
    private func accelLive(at time: Double) -> Bool { (useIns ? ins.ready : centre.ready) && time - lastAccelTime < 0.25 }

    /// The right palm (world), from the hand tracker, at its capture time, with the camera's position
    /// then. The tags and the IMU place the paddle; once no tag has been seen for a moment (0.1 s) the
    /// hand takes over where the camera sees it:
    /// - across the line of sight, firmly (±1.5 cm): the hand's direction in the image is precise, and
    ///   since everything is drawn as the camera sees it, that's where the paddle has to be to be in
    ///   the hand;
    /// - along it, only loosely (±10 cm): the hand's distance is guessed from its apparent size.
    /// The palm -> blade centre offset is learned while a tag and the hand are both seen.
    func addPalm(_ position: V3, time: Double, camera: V3? = nil) {
        lastPalm = (time, position)
        if let s = smoothPalm, time - s.time < 0.2, time > s.time {
            let k = 1 - exp(-(time - s.time) / 0.06)
            smoothPalm = (time, s.position + (position - s.position) * k)
        } else {
            smoothPalm = (time, position)
        }
        let dt = min(max(time - (previousPalmTime ?? time), 0), 0.05)
        previousPalmTime = time
        guard run == nil, grip.ready, let cal = calibration, let q = orientation(at: time, calibration: cal) else { return }
        let offset = palmToGrip ?? Self.defaultPalmToGrip
        if let tag = lastTag, time - tag.time < Self.handAfter { return }
        if accelLive(at: time), useIns {
            let target = position + q.act(offset - Self.gripFromCentre)      // the blade centre, by the hand
            let k = FusionParams.current
            let R = InsFilter.anisotropic(across: k.handAcross, along: k.handAlong, sight: camera.map { vnormalize(target - $0) })
            glide(-ins.correct(target, R: R, at: time), at: time)
        } else if accelLive(at: time), let then = centre.state(at: time) {
            let target = position + q.act(offset - Self.gripFromCentre)      // the blade centre, by the hand
            let r = target - then.p
            if let camera {
                let sight = vnormalize(target - camera)
                let along = sight * vdot(r, sight)
                glide(-centre.correct(then.p + (r - along), sigma: FusionParams.current.handAcross, at: time,
                                      gate: FusionParams.current.handGate), at: time)
                if let now = centre.state(at: time) {
                    let r2 = target - now.p
                    glide(-centre.correct(now.p + sight * vdot(r2, sight), sigma: FusionParams.current.handAlong, at: time,
                                          gate: FusionParams.current.handGate + 0.1), at: time)
                }
            } else {
                glide(-centre.correct(target, sigma: 0.05, at: time, gate: 0.5), at: time)
            }
        }
        let (now, _) = grip.predict(to: time)
        let error = smoothPalm!.position + q.act(offset) - now
        guard vlength(error) < 0.6 else { return }
        var step = error * (1 - exp(-dt / Self.palmPull))
        if vlength(step) > 0.015 { step = vnormalize(step) * 0.015 }      // never a visible jump
        grip.shift(by: step)
    }

    /// s without a tag before the hand takes over.
    static var handAfter: Double { FusionParams.current.handAfter }

    private var previousPalmTime: Double?
    /// Where the hand holds the handle (palm -> grip point, paddle frame) until it's learned.
    static let defaultPalmToGrip = V3(0, 0, -0.02)
    static let palmPull = 0.15             // s: how quickly the hand pulls a tagless paddle back

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
        // A clear view: its best fit. An ambiguous one: the candidate the IMU agrees with.
        var tag = sighting.candidates.min { $0.error < $1.error }!
        if !Self.isClear(sighting), let heading {
            let predicted = heading * b * cal.mountQuat
            tag = sighting.candidates.min { Quat.angle($0.rotation, predicted) < Quat.angle($1.rotation, predicted) }!
        }
        let paddle = (tag.rotation * tagFromPaddle).normalized
        let centre = tag.position + paddle.act(Self.tagToCentre)
        lastBigHeading = ((tag.rotation * cal.mountQuat.conjugate * b.conjugate).normalized, t)
        refine(with: sighting, calibration: cal, imu: b)
        fuse(paddle: paddle, centre: centre, time: t, imu: b, calibration: cal, clear: Self.isClear(sighting), tag: sighting.id)
    }

    /// A small tag on the forehand face. Its offset from the blade centre is known; how it is turned
    /// is learned (the first time both small tags are in view, the line from 69 to 68 is the paddle's
    /// x; or from the calibrated IMU), then kept.
    private func addSmall(_ sighting: PaddleTagSighting, offset: V3) {
        let t = sighting.time
        smallSeen.append((sighting.id, t, sighting.candidates))
        if smallSeen.count > 8 { smallSeen.removeFirst(smallSeen.count - 8) }
        let cal = calibration
        let b = cal.flatMap { imuOrientation(at: t + $0.timeOffset) }
        let imuPaddle: Quat? = (heading != nil && b != nil && cal != nil)
            ? heading! * b! * cal!.mountQuat * tagFromPaddle : nil

        guard let turns = smallTurns[sighting.id] else {
            learnSmall(sighting, imuPaddle: imuPaddle)
            return
        }
        let placement = Self.forehandPlacements[turns]
        // The planar ambiguity: the candidate the IMU agrees with (unless it agrees with neither: a wrong
        // big-tag turn, being fixed below), else the one nearest the last paddle.
        var tag = sighting.candidates.min { $0.error < $1.error }!
        let imuAgrees = imuPaddle.map { p in sighting.candidates.contains { Quat.angle($0.rotation * placement, p) < .pi / 4 } } ?? false
        let recentRaw = rawPaddle.flatMap { t - $0.time < 0.3 ? $0.rotation : nil }
        if !Self.isClear(sighting), let reference = imuAgrees ? imuPaddle : recentRaw ?? imuPaddle {
            tag = sighting.candidates.min {
                Quat.angle($0.rotation * placement, reference) < Quat.angle($1.rotation * placement, reference)
            }!
        }
        let paddle = (tag.rotation * placement).normalized
        let centre = tag.position - paddle.act(offset + (sideCorrection[sighting.id] ?? .zero))
        if let cal, let b {
            checkBigTagTurns(smallPaddles: sighting.candidates.map { ($0.rotation * placement).normalized }, imu: b,
                             calibration: cal, time: t)
        }
        setRawPaddle(time: t, rotation: paddle, centre: centre)
        if run == nil, let cal, let b {
            fuse(paddle: paddle, centre: centre, time: t, imu: b, calibration: cal, clear: Self.isClear(sighting),
                 tag: sighting.id)
        }
    }

    /// Keeps the calibration accurate: the big tag's recent views (every second one, the last ~25 s) are
    /// refitted every 10 s around the current clock offset, and a clearly different, good fit (< 2.5°,
    /// more than 1° away) replaces the mount. The paddle's orientation stays continuous across the switch.
    private func refine(with sighting: PaddleTagSighting, calibration cal: PaddleCalibration, imu b: Quat) {
        refineCounter += 1
        if refineCounter % 2 == 0 {
            refineViews.append((sighting.time, sighting.candidates.map(\.rotation), sighting.candidates[0].position))
            if refineViews.count > 300 { refineViews.removeFirst(refineViews.count - 300) }
        }
        guard refineViews.count >= 150, sighting.time - lastRefine > 10 else { return }
        lastRefine = sighting.time
        var r = CalibrationRun()
        r.sightings = refineViews
        guard let fit = solve(r, offsets: (cal.timeOffset - 0.006)...(cal.timeOffset + 0.006), gaps: [1, 3, 6, 10]),
              fit.residualDeg < min(2.5, cal.residualDeg * 0.8),        // clearly better: it settles, no flip-flop
              Quat.angle(fit.mountQuat, cal.mountQuat) > 1 * .pi / 180 else { return }
        var better = fit
        better.tagTurns = cal.tagTurns
        let keptHeading = heading.map { ($0 * b * cal.mountQuat * better.mountQuat.conjugate * b.conjugate).normalized }
        calibration = better
        heading = keptHeading
        onEvent?(String(format: "paddle: calibration refined, %.1f° moved, fit %.2f°, clock %.1f ms",
                        Quat.angle(fit.mountQuat, cal.mountQuat) * 180 / .pi, fit.residualDeg, fit.timeOffset * 1000))
        onCalibrationImproved?(better)
    }

    private var refineViews: [(time: Double, candidates: [Quat], position: V3)] = []
    private var refineCounter = 0
    private var lastRefine = -Double.infinity

    /// The small tags know the paddle's true orientation (from their layout); the big tag's turns are
    /// right when both sides give the IMU the same heading. If another quarter turn fits clearly better
    /// for a while, it becomes the calibration's.
    private func checkBigTagTurns(smallPaddles: [Quat], imu b: Quat, calibration cal: PaddleCalibration, time t: Double) {
        guard tagTurnsOverride == nil, let big = lastBigHeading, t - big.time < 3 else { return }
        let angles = (0..<4).map { k -> Double in
            let turn = Quat(rotationVector: V3(0, 0, Double(k) * .pi / 2))
            return smallPaddles.map {
                Quat.angle(($0 * (cal.mountQuat * turn).conjugate * b.conjugate).normalized, big.heading)
            }.min()!
        }
        guard let k = angles.indices.min(by: { angles[$0] < angles[$1] }), angles[k] < 20 * .pi / 180 else { return }
        bigTurnVotes[k] += 1
        let total = bigTurnVotes.reduce(0, +)
        guard bigTurnVotes[k] >= 6, Double(bigTurnVotes[k]) >= 0.8 * Double(total) else {
            if total > 30 { bigTurnVotes = [0, 0, 0, 0] }
            return
        }
        bigTurnVotes = [0, 0, 0, 0]
        guard k != cal.tagTurns else { return }
        var fixed = cal
        fixed.tagTurns = k
        calibration = fixed
        heading = nil                   // it was being pulled by the wrong turn: start it afresh
        headingOutliers = 0
        onEvent?("paddle: big tag turns corrected \(cal.tagTurns) -> \(k) (the small tags disagreed)")
        onCalibrationImproved?(fixed)
    }

    private func learnSmall(_ sighting: PaddleTagSighting, imuPaddle: Quat?) {
        // Both in this frame: the paddle's x points from 69 to 68. Each unlearned one votes for the
        // quarter turn that lines its own x up with that.
        let t = sighting.time
        guard let other = smallSeen.first(where: { $0.id != sighting.id && abs($0.time - t) < 0.002 }) else { return }
        let mine = sighting.candidates[0].position, theirs = other.candidates[0].position
        guard abs(vlength(mine - theirs) - 0.11) < 0.03 else { return }
        let x = vnormalize(sighting.id == 68 ? mine - theirs : theirs - mine)
        for (id, candidates) in [(sighting.id, sighting.candidates), (other.id, other.candidates)] where smallTurns[id] == nil {
            var best = (score: -2.0, k: 0)
            for c in candidates {
                for k in 0..<4 {
                    let score = vdot((c.rotation * Self.forehandPlacements[k]).act(V3(1, 0, 0)), x)
                    if score > best.score { best = (score, k) }
                }
            }
            guard best.score > 0.85 else { continue }
            smallVotes[id, default: [0, 0, 0, 0]][best.k] += 1
            let votes = smallVotes[id]!
            let total = votes.reduce(0, +)
            if total >= 10, let k = votes.indices.max(by: { votes[$0] < votes[$1] }), Double(votes[k]) >= 0.7 * Double(total) {
                smallTurns[id] = k
                onEvent?("paddle: tag \(id) placement learned (\(k) quarter turns, \(votes))")
                onSmallTagsLearned?(smallTurns)
            }
        }
    }

    /// One pose clearly fits better than the other (a flat tag seen nearly face-on fits two about
    /// equally well): only such a view may start or restart the heading.
    private static func isClear(_ sighting: PaddleTagSighting) -> Bool {
        let errors = sighting.candidates.map(\.error).sorted()
        return errors.count < 2 || errors[0] * 5 < errors[1]
    }

    /// One paddle pose seen by a tag: keeps the IMU heading and the grip point.
    private func fuse(paddle: Quat, centre centre_: V3, time t: Double, imu b: Quat, calibration cal: PaddleCalibration,
                      clear: Bool, tag id: Int) {
        // Heading: the IMU's earth frame in the world, smoothed (1/2 s), restarted after a run of
        // disagreeing sightings (the paddle's board restarted, say).
        let observed = (paddle * (cal.mountQuat * tagFromPaddle).conjugate * b.conjugate).normalized
        if let current = heading {
            if Quat.angle(current, observed) > 20 * .pi / 180 {
                if clear { headingOutliers += 1 }
                if headingOutliers >= 6 {
                    heading = observed
                    headingOutliers = 0
                    onEvent?("paddle: heading re-acquired")
                }
            } else {
                headingOutliers = 0
                let alpha = 1 - exp(-min(max(t - headingTime, 0), 0.5) / FusionParams.current.headingTau)
                heading = current.slerp(observed, alpha)
            }
        } else {
            guard clear else { return }
            heading = observed
            onEvent?("paddle: heading acquired")
        }
        headingTime = t

        // Grip point from the tag (its own rotation is exact at this instant). After a gap the tags put
        // it back where it really is; the shown paddle glides there instead of jumping.
        let gripPoint = centre_ + paddle.act(Self.gripFromCentre)
        let was = grip.ready ? grip.predict(to: t).0 : nil
        let sinceLast = t - grip.time
        let sideChanged = lastFusedTag.map { Self.bigTagIDs.contains($0) != Self.bigTagIDs.contains(id) } ?? false
        // The paddle just turned to its other side: where the IMU and the last side put the grip vs where
        // this side's tag does is (mostly) this small tag's placement error. Learn it, a little each time.
        if sideChanged, sinceLast < 0.25, let was {
            let small = Self.bigTagIDs.contains(id) ? lastFusedTag! : id
            let sign: Double = Self.bigTagIDs.contains(id) ? -1 : 1     // which side's view is the small tag's
            let residual = paddle.conjugate.act((was - gripPoint) * sign)
            var c = (sideCorrection[small] ?? .zero) + residual * 0.25
            if vlength(c) > 0.06 { c = vnormalize(c) * 0.06 }
            sideCorrection[small] = c
        }
        lastFusedTag = id
        if useIns {
            if ins.ready, accelLive(at: t) {
                let sigma = FusionParams.current.tagSigma
                glide(-ins.correct(centre_, R: Mat.identity(3, sigma * sigma), at: t), at: t)
            } else {
                ins.reset(at: t, position: centre_)
            }
        } else if centre.ready, accelLive(at: t) {
            glide(-centre.correct(centre_, sigma: FusionParams.current.tagSigma, at: t), at: t)
        } else {
            centre.reset(at: t, position: centre_)
        }
        grip.update(gripPoint, time: t)
        if let was {
            // A correction of more than a centimetre (the tags back after a gap, the other side's tags)
            // glides in over ~0.1 s: the shown paddle moves there instead of jumping.
            if !accelLive(at: t) { glide(was - grip.predict(to: t).0, at: t) }
        }
        // Learn where the hand holds the handle, for when no tag is seen.
        if let palm = smoothPalm, abs(palm.time - t) < 0.04 {
            let offset = paddle.conjugate.act(gripPoint - palm.position)
            if vlength(offset) < 0.25 { palmToGrip = palmToGrip.map { $0 + (offset - $0) * 0.05 } ?? offset }
        }
        lastTag = (t, paddle, centre_)
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
        if shownOffsetTime == nil { shownOffsetTime = useFrameClock ? frameClock : time }   // glides from the frame it's first shown
        let omega = imuToWorld.act(rate)
        var state: PaddleState
        if accelLive(at: time) {
            // The accelerometer's blade centre (the IMU is at it), looked ahead to `time`.
            let (c, v) = useIns ? ins.predict(to: time) : centre.predict(to: time)
            state = PaddleState(center: c + shownOffsetNow(at: time), basis: paddle.basis)
            state.velocity = v
        } else {
            // No accelerometer: turning about the grip, which the tags (and the hand) place.
            var (g, gv) = grip.predict(to: time)
            g += shownOffsetNow(at: time)
            let c = g - paddle.act(Self.gripFromCentre)
            state = PaddleState(center: c, basis: paddle.basis)
            state.velocity = gv + vcross(omega, c - g)
        }
        state.angular = omega
        state.visible = true
        let tagRecent = lastTag.map { time - $0.time < 0.25 } ?? false
        let handRecent = lastPalm.map { time - $0.time < 0.25 } ?? false
        source = tagRecent ? "imu+tag" : handRecent ? "imu+hand" : "imu (holding position)"
        return state
    }

    /// A correction of the position by -`step`: shown gliding over ~0.1 s if it's more than a centimetre.
    private func glide(_ step: V3, at t: Double) {
        guard vlength(step) > Self.glideFrom else { return }
        let (x, v) = shownOffsetState(at: useFrameClock ? frameClock! : t)
        shownOffset = x + step
        shownRate = v
        shownOffsetTime = nil                           // glides from the frame it's first shown
    }

    private func shownOffsetNow(at time: Double) -> V3 { shownOffsetState(at: time).0 }
    /// The paddle to draw, one frame after another (`time` increasing): `pose(at:)` with its position
    /// through a one-euro filter (the research's render-only smoothing). It smooths the small shakes the
    /// tag and hand corrections leave while the paddle is slow, and lets fast swings through at once.
    /// The physics keeps the unsmoothed `pose(at:)`.
    func drawnPose(at time: Double) -> PaddleState? {
        if let d = drawn, d.time == time { return d.pose }
        guard var p = pose(at: time) else { drawn = nil; return nil }
        let k = FusionParams.current
        if k.euroMin > 0, let d = drawn, time > d.time, time - d.time < 0.1 {
            let dt = time - d.time
            let cutoff = k.euroMin + k.euroBeta * vlength(p.velocity)        // Hz
            let a = 1 / (1 + 1 / (2 * .pi * cutoff * dt))
            p.center = d.pose.center + (p.center - d.pose.center) * a
        }
        drawn = (time, p)
        return p
    }
    private var drawn: (time: Double, pose: PaddleState)?

    /// The correction still easing out at `time` (for the replay's diagnostics).
    func offsetForDebug(at time: Double) -> V3 { shownOffsetNow(at: time) }

    /// The shown offset and its rate: easing out exponentially, or (glideShape 1) critically damped, which
    /// starts the ease without a kink in the shown paddle's speed and keeps an ongoing ease's speed.
    private func shownOffsetState(at query: Double) -> (V3, V3) {
        guard let start = shownOffsetTime else { return (shownOffset, shownRate) }
        let time = useFrameClock ? frameClock! : query
        let dt = max(time - start, 0)
        if FusionParams.current.glideShape == 1 {
            let w = 2 / Self.glide
            let b = shownRate + shownOffset * w
            let e = exp(-w * dt)
            return ((shownOffset + b * dt) * e, (shownRate - b * (w * dt)) * e)
        }
        let e = exp(-dt / Self.glide)
        return (shownOffset * e, shownOffset * (-e / Self.glide))
    }

    /// How long a jump of the shown paddle takes to ease out (s).
    static var glide: Double { FusionParams.current.glide }
    /// Corrections smaller than this apply at once (no lag while the tags are seen).
    static var glideFrom: Double { FusionParams.current.glideFrom }

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
        if r.sightings.count < 40 { progress?.fraction = 0.5 * Double(r.sightings.count) / 40 }
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
        frameClock = time
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
        return min(p.fraction, 0.97)
    }

    /// Why the calibration isn't done yet: shown by the ring, the HUD, and the log (when it changes).
    private func blocked(_ message: String, fraction: Double) {
        guard progress != nil else { return }               // a background refinement: no messages
        if progress?.message != message { onEvent?("paddle: calibrating: \(message)") }
        progress?.message = message
        progress?.fraction = fraction
    }

    /// The mount and clock offset that best explain the tag's rotations by the IMU's.
    private func solve(_ r: CalibrationRun, offsets: ClosedRange<Double> = -0.060...0.040,
                       gaps: [Int] = [2, 4, 7, 11, 16, 24]) -> PaddleCalibration? {
        let s = r.sightings
        // Pairs of views a moderate turn apart, with the relative rotation (tag frame) for each
        // choice of candidates.
        var pairs: [(i: Int, j: Int, options: [V3])] = []
        for i in s.indices {
            for gap in gaps where i + gap < s.count {
                let j = i + gap
                guard s[j].time - s[i].time < 1.5 else { continue }
                var options: [V3] = []
                for a in s[i].candidates { for b in s[j].candidates { options.append((a.conjugate * b).rotationVector) } }
                if options.contains(where: { vlength($0) > 0.17 && vlength($0) < 2.1 }) { pairs.append((i, j, options)) }
            }
        }
        guard pairs.count >= 30 else {
            blocked("turn the paddle more", fraction: 0.5)
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
        var offset = offsets.lowerBound
        while offset <= offsets.upperBound + 1e-9 {
            if let f = fit(offset: offset), f.residual < best?.residual ?? .infinity { best = (offset, f.mount, f.residual, f.data) }
            offset += 0.002
        }
        guard let coarse = best else {
            blocked("the tag's turns don't match the IMU's yet: keep turning slowly", fraction: 0.5)
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
        let axes = symmetricEigen(spread).values.sorted()[1]
        guard axes > 0.08 else {
            blocked(String(format: "tilt it the other way too (forward/back and side to side), axes %.2f of 0.08", axes),
                    fraction: 0.5 + 0.35 * min(axes / 0.08, 1))
            return nil
        }
        // The MPU6050's gyro scale is off by a few %, so the fit can't be much tighter than this.
        let residualDeg = result.residual * 180 / .pi
        guard residualDeg < 4.0 else {
            blocked(String(format: "fit %.1f° (needs < 4°): keep turning slowly", residualDeg),
                    fraction: 0.85 + 0.1 * max(0, 1 - (residualDeg - 4) / 6))
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

/// The blade centre between tag sightings: position and velocity driven by the accelerometer, corrected
/// by position measurements (a Kalman filter, the same variances on all three axes). Measurements come
/// from camera frames, so they are late: each is compared with the state at its capture time, and the
/// correction carried forward to now (exact for this linear model). Keeps ~0.75 s of states for that.
struct CentreFilter {
    struct State {
        var time: Double
        var p: V3
        var v: V3
    }

    private(set) var now: State?
    private var history: [State] = []
    private var pp = 1.0, pv = 0.0, vv = 1.0
    /// Accelerometer error, (m/s^2)^2 s: its noise, bias and the gravity leaking through small
    /// orientation errors.
    static var accelNoise: Double { FusionParams.current.accelNoise }

    var ready: Bool { now != nil }

    mutating func reset(at time: Double, position: V3) {
        let s = State(time: time, p: position, v: .zero)
        now = s
        history = [s]
        pp = 1e-5; pv = 0; vv = 0.25
    }

    /// On to `time` with the world acceleration `a` (m/s^2); `damping` (1/s) leaks the velocity away.
    mutating func propagate(to time: Double, accel a: V3, damping: Double, noise: Double = 1) {
        guard var s = now, time > s.time else { return }
        let dt = min(time - s.time, 0.05)
        s.p += s.v * dt + a * (0.5 * dt * dt)
        s.v = (s.v + a * dt) * exp(-damping * dt)
        if vlength(s.v) > 8 { s.v = vnormalize(s.v) * 8 }
        s.time = time
        let q = Self.accelNoise * noise
        let npp = pp + 2 * dt * pv + dt * dt * vv + q * dt * dt * dt / 3
        let npv = pv + dt * vv + q * dt * dt / 2
        let nvv = vv + q * dt
        pp = npp; pv = npv; vv = nvv
        now = s
        history.append(s)
        if history.count > 150 { history.removeFirst(history.count - 150) }
    }

    /// The blade centre measured at `time` (`sigma` m): returns how far it moved the current position.
    mutating func correct(_ z: V3, sigma: Double, at time: Double, gate: Double = .infinity) -> V3 {
        guard let s = now else { return .zero }
        let then = state(at: time) ?? s
        let r = z - then.p
        guard vlength(r) < gate else { return .zero }
        let total = pp + sigma * sigma
        let kp = pp / total, kv = pv / total
        let npp = (1 - kp) * pp, npv = (1 - kp) * pv, nvv = vv - kv * pv
        pp = npp; pv = npv; vv = nvv
        for i in history.indices where history[i].time >= time {
            history[i].p += r * kp + r * (kv * (history[i].time - time))
            history[i].v += r * kv
        }
        let shift = r * kp + r * (kv * max(s.time - time, 0))
        now!.p += shift
        now!.v += r * kv
        return shift
    }

    /// The state at a past `time` (interpolated), or nil if it's older than the history.
    func state(at time: Double) -> State? {
        guard let last = history.last else { return nil }
        if time >= last.time { return last }
        guard let i = history.lastIndex(where: { $0.time <= time }) else { return nil }
        let a = history[i], b = history[min(i + 1, history.count - 1)]
        let f = b.time > a.time ? (time - a.time) / (b.time - a.time) : 0
        return State(time: time, p: a.p + (b.p - a.p) * f, v: a.v + (b.v - a.v) * f)
    }

    /// Position and velocity at `time`: from the history if it's past, else looked ahead from now (at
    /// most 0.1 s).
    func predict(to time: Double) -> (V3, V3) {
        guard let s = now else { return (.zero, .zero) }
        if time < s.time, let past = state(at: time) { return (past.p, past.v) }
        let dt = min(max(time - s.time, 0), FusionParams.current.lookAhead)
        return (s.p + s.v * dt, s.v)
    }
}

/// The grip point (middle of the handle) from the tags: an alpha-beta filter. Its velocity, for looking
/// ahead to the display time, is smoothed over several sightings: the difference of two sightings is
/// mostly their noise, and looking ahead with it made the paddle saw back and forth by centimetres.
/// Without tags the velocity fades and the grip holds (the IMU still turns the paddle about it).
struct PivotTracker {
    static let gap = 0.3                    // s without a sighting: start over from the next one
    private(set) var position = V3.zero
    private(set) var velocity = V3.zero
    private(set) var time = 0.0
    private(set) var ready = false
    static let alpha = 0.6, beta = 0.06
    static let fade = 0.25                  // s: how long a lost paddle keeps its velocity
    static let horizon = 0.3                // s: the furthest it is looked ahead

    mutating func update(_ z: V3, time t: Double) {
        guard ready, t - time < Self.gap, t > time else {
            if !ready || t - time >= Self.gap {
                position = z
                velocity = .zero
                time = t
                ready = true
            }
            return
        }
        let dt = t - time
        let predicted = position + velocity * dt
        let r = z - predicted
        position = predicted + r * Self.alpha
        velocity += r * (Self.beta / dt)
        if vlength(velocity) > 4 { velocity = vnormalize(velocity) * 4 }
        time = t
    }

    /// Moved without a sighting (the hand carried it).
    mutating func shift(by d: V3) { position += d }

    /// Position and velocity at `t`: looked ahead with the velocity, which fades once the tags are gone.
    func predict(to t: Double) -> (V3, V3) {
        let dt = min(max(t - time, 0), Self.horizon)
        let fade = exp(-max(t - time - 0.05, 0) / Self.fade)
        return (position + velocity * (dt * fade), velocity * fade)
    }
}


/// The fusion's tuning, in one place: the defaults are what the paddle replay (scripts/paddle-replay)
/// scored best on real recordings; the replay sets others to compare.
/// The fusion's tuning. Defaults: tuned on recorded play with scripts/paddle-replay (tags hidden 0.6 s of
/// every 2 s; a 3 min recording replayed frame by frame as the app drew it): the bias-estimating filter,
/// the arm's reach and the drawn paddle's one-euro smoothing.
struct FusionParams {
    var accelNoise = 0.25               // (m/s^2)^2 s: accelerometer error in the position filter
    var tagSigma = 0.0057               // m: a tag's blade-centre measurement
    var damping = 0.5                   // 1/s: velocity leak once no tag for `dampAfter`
    var dampAfter = 0.0                 // s
    var stillDamping = 8.0              // 1/s: when the IMU says the paddle is held still
    var handAfter = 0.27                // s without a tag before the hand corrects
    var handAcross = 0.049              // m: the hand across the line of sight (its image position)
    var handAlong = 0.16                // m: along it (its distance, from its apparent size)
    var handGate = 0.4                  // m: a hand reading further off than this is ignored
    var imuFromCentre = V3(0.028, 0.06, 0) // the IMU from the blade centre, IMU frame
    var glide = 0.17                    // s: how long a correction takes to show
    var glideFrom = 0.008               // m: smaller corrections show at once
    var lookAhead = 0.1                 // s: the furthest the position is extrapolated
    var headingTau = 0.6                // s: heading smoothing
    var filter = 1                      // position filter: 0 position/velocity, 1 with accelerometer bias (InsFilter)
    var biasWalk = 0.01                 // g/sqrt(s): how fast the accelerometer's bias may wander (filter 1)
    var gate = 900.0                    // chi^2 (3 dof): measurements further off are ignored (filter 1; tuned: only wild ones)
    var armReach = 0.55                 // m: shoulder -> blade centre at most (0: off)
    var armSigma = 0.08                 // m: how firmly the reach holds
    var motionA = 0.0                   // accelerometer noise x (1 + (motionA (|f| - 1 g))^2 + (motionW |w|)^2)
    var motionW = 0.0
    var glideShape = 0                  // 0 exponential, 1 critically damped
    var euroMin = 4.0                   // Hz: the drawn paddle's smoothing when still (0: off)
    var euroBeta = 5.0                  // Hz per m/s: opened up with speed
    var glideClock = 1                  // corrections ease out on: 0 the asked time, 1 the frame's time

    static var current = FusionParams()

    /// "name=value,name=value" (the replay's PARAMS).
    mutating func set(_ text: String) {
        for item in text.split(separator: ",") {
            let kv = item.split(separator: "=")
            guard kv.count == 2, let v = Double(kv[1]) else { continue }
            switch kv[0] {
            case "accelNoise": accelNoise = v
            case "tagSigma": tagSigma = v
            case "damping": damping = v
            case "dampAfter": dampAfter = v
            case "stillDamping": stillDamping = v
            case "handAfter": handAfter = v
            case "handAcross": handAcross = v
            case "handAlong": handAlong = v
            case "handGate": handGate = v
            case "imuX": imuFromCentre.x = v
            case "imuY": imuFromCentre.y = v
            case "imuZ": imuFromCentre.z = v
            case "glide": glide = v
            case "glideFrom": glideFrom = v
            case "lookAhead": lookAhead = v
            case "headingTau": headingTau = v
            case "filter": filter = Int(v)
            case "biasWalk": biasWalk = v
            case "gate": gate = v
            case "armReach": armReach = v
            case "armSigma": armSigma = v
            case "motionA": motionA = v
            case "motionW": motionW = v
            case "glideShape": glideShape = Int(v)
            case "glideClock": glideClock = Int(v)
            case "euroMin": euroMin = v
            case "euroBeta": euroBeta = v
            default: print("unknown parameter \(kv[0])")
            }
        }
    }
}
