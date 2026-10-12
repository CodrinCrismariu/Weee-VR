import Foundation

/// A small dense matrix (row-major), for the paddle's position filter.
struct Mat {
    var rows: Int, cols: Int
    var a: [Double]

    init(_ rows: Int, _ cols: Int, _ value: Double = 0) {
        self.rows = rows; self.cols = cols
        a = [Double](repeating: value, count: rows * cols)
    }

    static func identity(_ n: Int, _ s: Double = 1) -> Mat {
        var m = Mat(n, n)
        for i in 0..<n { m[i, i] = s }
        return m
    }

    subscript(_ i: Int, _ j: Int) -> Double {
        get { a[i * cols + j] }
        set { a[i * cols + j] = newValue }
    }

    static func * (l: Mat, r: Mat) -> Mat {
        var m = Mat(l.rows, r.cols)
        for i in 0..<l.rows {
            for k in 0..<l.cols {
                let v = l[i, k]
                if v == 0 { continue }
                for j in 0..<r.cols { m.a[i * r.cols + j] += v * r.a[k * r.cols + j] }
            }
        }
        return m
    }

    static func + (l: Mat, r: Mat) -> Mat { var m = l; for i in m.a.indices { m.a[i] += r.a[i] }; return m }
    static func - (l: Mat, r: Mat) -> Mat { var m = l; for i in m.a.indices { m.a[i] -= r.a[i] }; return m }

    var t: Mat {
        var m = Mat(cols, rows)
        for i in 0..<rows { for j in 0..<cols { m[j, i] = self[i, j] } }
        return m
    }

    /// A 3x3 block at (r, c), from the columns of a basis (rotation) times `s`.
    mutating func setBlock(_ r: Int, _ c: Int, _ b: Basis, _ s: Double) {
        let cols = [b.x, b.y, b.z]
        for j in 0..<3 { self[r, c + j] = cols[j].x * s; self[r + 1, c + j] = cols[j].y * s; self[r + 2, c + j] = cols[j].z * s }
    }

    /// The inverse of a 3x3.
    var inverse3: Mat {
        let m = self
        let det = m[0, 0] * (m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1]) - m[0, 1] * (m[1, 0] * m[2, 2] - m[1, 2] * m[2, 0])
            + m[0, 2] * (m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0])
        var r = Mat(3, 3)
        guard abs(det) > 1e-30 else { return Mat.identity(3) }
        r[0, 0] = (m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1]) / det
        r[0, 1] = (m[0, 2] * m[2, 1] - m[0, 1] * m[2, 2]) / det
        r[0, 2] = (m[0, 1] * m[1, 2] - m[0, 2] * m[1, 1]) / det
        r[1, 0] = (m[1, 2] * m[2, 0] - m[1, 0] * m[2, 2]) / det
        r[1, 1] = (m[0, 0] * m[2, 2] - m[0, 2] * m[2, 0]) / det
        r[1, 2] = (m[0, 2] * m[1, 0] - m[0, 0] * m[1, 2]) / det
        r[2, 0] = (m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0]) / det
        r[2, 1] = (m[0, 1] * m[2, 0] - m[0, 0] * m[2, 1]) / det
        r[2, 2] = (m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]) / det
        return r
    }
}

/// The blade centre with the accelerometer's bias (the research's "error-state" recommendation, for
/// position): state [position, velocity (world), accelerometer bias (IMU frame, g)], 9 states.
/// - Each IMU sample propagates it: a = heading (B (f - bias) - up) g, with B the IMU's own orientation.
///   The bias is learned while tags are seen and the paddle turns (it shows up in different world
///   directions as it turns), so it no longer drifts the position in a gap.
/// - Late measurements (camera frames) rewind to their capture time and replay the samples since:
///   exact, unlike carrying a correction forward.
/// - Position measurements may be anisotropic (the hand: firm across the line of sight, loose along it),
///   and are gated by their Mahalanobis distance (99%, 3 dof) with a Huber weight past 1.5 sigma.
struct InsFilter {
    struct Input {
        var time: Double
        var specific: V3            // g, IMU frame (lever arm already removed)
        var imuToEarth: Quat        // B
        var heading: Quat           // IMU earth -> world
        var damping: Double         // 1/s
        var noise: Double = 1       // accelerometer noise scale (hard swings)
    }

    private struct Step {
        var time: Double
        var x: [Double]             // p v b
        var P: Mat
        var input: Input?           // what brought it here (nil: a reset)
    }

    private var history: [Step] = []
    var ready: Bool { !history.isEmpty }
    var time: Double { history.last?.time ?? 0 }

    static var accelNoise: Double { FusionParams.current.accelNoise }          // (m/s^2)^2 s
    static var biasWalk: Double { FusionParams.current.biasWalk }              // g / sqrt(s)
    static var gate: Double { FusionParams.current.gate }                      // chi^2, 3 dof

    mutating func reset(at time: Double, position: V3) {
        var P = Mat.identity(9, 1e-6)
        for i in 3..<6 { P[i, i] = 0.25 }
        for i in 6..<9 { P[i, i] = 0.03 * 0.03 }
        let bias = history.last.map { Array($0.x[6..<9]) } ?? [0, 0, 0]       // the bias survives a reset
        history = [Step(time: time, x: [position.x, position.y, position.z, 0, 0, 0] + bias, P: P, input: nil)]
    }

    mutating func propagate(_ input: Input) {
        guard let last = history.last, input.time > last.time else { return }
        history.append(Self.step(last, input))
        if history.count > 120 { history.removeFirst(history.count - 120) }
    }

    private static func step(_ s: Step, _ u: Input) -> Step {
        let dt = min(u.time - s.time, 0.05)
        let g = PaddleFusion.gravity
        let p = V3(s.x[0], s.x[1], s.x[2]), v = V3(s.x[3], s.x[4], s.x[5]), b = V3(s.x[6], s.x[7], s.x[8])
        let a = u.heading.act(u.imuToEarth.act(u.specific - b) - V3(0, 0, 1)) * g
        let np = p + v * dt + a * (0.5 * dt * dt)
        var nv = v + a * dt
        nv = nv * exp(-u.damping * dt)
        if vlength(nv) > 8 { nv = vnormalize(nv) * 8 }
        // Error transition: p <- p + v dt - 1/2 M g dt^2 b, v <- v - M g dt b (M: IMU -> world).
        let M = (u.heading * u.imuToEarth).basis
        var F = Mat.identity(9)
        for i in 0..<3 { F[i, 3 + i] = dt }
        F.setBlock(0, 6, M, -0.5 * g * dt * dt)
        F.setBlock(3, 6, M, -g * dt)
        let q = accelNoise * u.noise, w = biasWalk * biasWalk * dt
        var Q = Mat(9, 9)
        for i in 0..<3 {
            Q[i, i] = q * dt * dt * dt / 3
            Q[i, 3 + i] = q * dt * dt / 2; Q[3 + i, i] = q * dt * dt / 2
            Q[3 + i, 3 + i] = q * dt
            Q[6 + i, 6 + i] = w
        }
        let P = F * s.P * F.t + Q
        return Step(time: u.time, x: [np.x, np.y, np.z, nv.x, nv.y, nv.z, b.x, b.y, b.z], P: P, input: u)
    }

    /// The blade centre measured at `time` with covariance `R` (3x3): rewinds to `time`, updates, replays.
    /// Returns how far it moved the current position (zero if gated out).
    @discardableResult
    mutating func correct(_ z: V3, R: Mat, at time: Double) -> V3 {
        guard let now = history.last else { return .zero }
        // The step at or just before the measurement.
        guard let k = history.lastIndex(where: { $0.time <= time + 1e-6 }) else { return .zero }
        var s = history[k]
        let r = [z.x - s.x[0], z.y - s.x[1], z.z - s.x[2]]
        var S = Mat(3, 3)
        for i in 0..<3 { for j in 0..<3 { S[i, j] = s.P[i, j] + R[i, j] } }
        let Si = S.inverse3
        var d2 = 0.0
        for i in 0..<3 { for j in 0..<3 { d2 += r[i] * Si[i, j] * r[j] } }
        guard d2 < Self.gate else { return .zero }
        // Huber: past 1.5 sigma the measurement counts less.
        let w = sqrt(d2) > 1.5 ? 1.5 / sqrt(d2) : 1
        var Rw = R
        if w < 1 { for i in Rw.a.indices { Rw.a[i] /= w }
            for i in 0..<3 { for j in 0..<3 { S[i, j] = s.P[i, j] + Rw[i, j] } } }
        let Sinv = S.inverse3
        // K = P H^T S^-1 (H picks the position).
        var PHt = Mat(9, 3)
        for i in 0..<9 { for j in 0..<3 { PHt[i, j] = s.P[i, j] } }
        let K = PHt * Sinv
        for i in 0..<9 { s.x[i] += K[i, 0] * r[0] + K[i, 1] * r[1] + K[i, 2] * r[2] }
        var KH = Mat(9, 9)
        for i in 0..<9 { for j in 0..<3 { KH[i, j] = K[i, j] } }
        s.P = (Mat.identity(9) - KH) * s.P
        // Replay what came after.
        history[k] = s
        var i = k + 1
        while i < history.count {
            if let u = history[i].input { history[i] = Self.step(history[i - 1], u) }
            i += 1
        }
        let after = history.last!
        return V3(after.x[0] - now.x[0], after.x[1] - now.x[1], after.x[2] - now.x[2])
    }

    /// Position and velocity at a past `time` (interpolated), or the newest.
    func state(at time: Double) -> (p: V3, v: V3)? {
        guard let last = history.last else { return nil }
        if time >= last.time { return (V3(last.x[0], last.x[1], last.x[2]), V3(last.x[3], last.x[4], last.x[5])) }
        guard let i = history.lastIndex(where: { $0.time <= time }) else { return nil }
        let a = history[i], b = history[min(i + 1, history.count - 1)]
        let f = b.time > a.time ? (time - a.time) / (b.time - a.time) : 0
        func v3(_ x: [Double], _ o: Int) -> V3 { V3(x[o], x[o + 1], x[o + 2]) }
        return (v3(a.x, 0) + (v3(b.x, 0) - v3(a.x, 0)) * f, v3(a.x, 3) + (v3(b.x, 3) - v3(a.x, 3)) * f)
    }

    /// At `time`: from the history if past, else looked ahead from now (at most `lookAhead`).
    func predict(to time: Double) -> (V3, V3) {
        guard let s = history.last else { return (.zero, .zero) }
        if time < s.time, let past = state(at: time) { return (past.p, past.v) }
        let dt = min(max(time - s.time, 0), FusionParams.current.lookAhead)
        let p = V3(s.x[0], s.x[1], s.x[2]), v = V3(s.x[3], s.x[4], s.x[5])
        return (p + v * dt, v)
    }

    var bias: V3 { history.last.map { V3($0.x[6], $0.x[7], $0.x[8]) } ?? .zero }

    /// Position covariance for a measurement: `across` firm perpendicular to `sight`, `along` on it.
    static func anisotropic(across: Double, along: Double, sight: V3?) -> Mat {
        var R = Mat.identity(3, across * across)
        guard let u = sight else { return R }
        let ua = [u.x, u.y, u.z]
        for i in 0..<3 { for j in 0..<3 { R[i, j] += (along * along - across * across) * ua[i] * ua[j] } }
        return R
    }
}
