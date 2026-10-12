// Replays smart-paddle recordings (the app's Settings > Record paddle data, Documents/paddle-rec/*.csv)
// through the app's own PaddleFusion at the render loop's 60 Hz and scores what the player would see.
//
//   scripts/paddle-replay/run.sh rec.csv [rec.csv ...]     PARAMS="accelNoise=4,handAcross=0.02" changes the tuning
//
// Each recording runs as it is, and again with the tags hidden from the fusion for 0.6 s of every 2 s
// at four phases (still scored against them): how well the paddle holds up without tags. Scores:
// - with tags / hidden: the shown blade centre vs each tag sighting (p50, p90, mm);
// - jumps: frames where the shown paddle moves much more than its own motion explains.
// SCORE = hidden p50 + 0.5 hidden p90 + with-tags p50 + 0.3 with-tags p90 + 3 jumps per 20 s (lower is better).
import Foundation

struct Record {
    enum Kind {
        case imu(PaddleIMUSample)
        case tag(Int, [PaddleTagSighting.Candidate], camera: V3)
        case palm(V3, camera: V3?)
        case shown(V3, head: V3)            // the paddle as the app drew it (captured at `time`)
    }
    var time: Double
    var kind: Kind
}

func load(_ path: String) -> (PaddleCalibration?, [Record]) {
    let lines = try! String(contentsOfFile: path, encoding: .utf8).split(separator: "\n")
    var calibration: PaddleCalibration?
    var records: [Record] = []
    for line in lines {
        let f = line.split(separator: ",").map(String.init)
        switch f.first {
        case "C":
            if f.count >= 7, let w = Double(f[1]) {
                calibration = PaddleCalibration(mount: [w, Double(f[2])!, Double(f[3])!, Double(f[4])!],
                                                timeOffset: Double(f[5])!, tagTurns: Int(f[6])!, residualDeg: 0)
            }
        case "I":
            let v = f.dropFirst().map { Double($0)! }
            records.append(Record(time: v[0], kind: .imu(PaddleIMUSample(
                time: v[0], orientation: Quat(w: v[1], x: v[2], y: v[3], z: v[4]), rate: V3(v[5], v[6], v[7]),
                still: v[8] != 0, accel: v.count >= 12 ? V3(v[9], v[10], v[11]) : nil))))
        case "T":
            let v = f.dropFirst().map { Double($0)! }
            let cam = Basis(x: V3(v[2], v[3], v[4]), y: V3(v[5], v[6], v[7]), z: V3(v[8], v[9], v[10]))
            let cpos = V3(v[11], v[12], v[13])
            var candidates: [PaddleTagSighting.Candidate] = []
            for k in 0..<Int(v[14]) {
                let b = 15 + 13 * k
                let r = Basis(x: V3(v[b], v[b + 1], v[b + 2]), y: V3(v[b + 3], v[b + 4], v[b + 5]), z: V3(v[b + 6], v[b + 7], v[b + 8]))
                let world = Basis(x: cam.apply(r.x), y: cam.apply(r.y), z: cam.apply(r.z))
                candidates.append(.init(rotation: Quat(basis: world), position: cpos + cam.apply(V3(v[b + 9], v[b + 10], v[b + 11])),
                                        error: v[b + 12]))
            }
            records.append(Record(time: v[0], kind: .tag(Int(v[1]), candidates, camera: cpos)))
        case "S":
            let v = f.dropFirst().map { Double($0)! }
            records.append(Record(time: v[0], kind: .shown(V3(v[1], v[2], v[3]), head: V3(v[8], v[9], v[10]))))
        case "P":
            let v = f.dropFirst().map { Double($0)! }
            records.append(Record(time: v[0], kind: .palm(V3(v[1], v[2], v[3]), camera: v.count >= 7 ? V3(v[4], v[5], v[6]) : nil)))
        default:
            break
        }
    }
    return (calibration, records)
}

struct Result {
    var visible: [Double] = [], hidden: [Double] = [], visibleDeg: [Double] = []
    var jumps = 0
    var seconds = 0.0
    var missing = 0
    var tableSpread: [Double] = []
    var kinks = 0                           // frames whose shown step differs > 20 mm from the last one
    var vsApp: [Double] = []                // the replay's drawn paddle vs the app's (same settings: ~0)
}

// How late each input reaches the render thread in the app.
let tagDelay = 0.025, palmDelay = 0.035

func run(_ calibration: PaddleCalibration?, _ records: [Record], drop: (period: Double, length: Double, phase: Double)?) -> Result {
    let fusion = PaddleFusion()
    fusion.autoCalibrate = false
    fusion.calibration = calibration
    if ProcessInfo.processInfo.environment["EVENTS"] != nil { fusion.onEvent = { print("  event: \($0)") } }
    var table = TableAnchor()
    let first = records.first!.time
    func hidden(_ t: Double) -> Bool {
        guard let d = drop, t - first > 1 else { return false }
        return (t - first - 1 + d.phase).truncatingRemainder(dividingBy: d.period) < d.length
    }
    // Events in arrival order.
    var events: [(arrival: Double, record: Record)] = records.map { r in
        switch r.kind {
        case .imu: return (r.time, r)
        case .tag: return (r.time + tagDelay, r)
        case .palm: return (r.time + palmDelay, r)
        case .shown: return (r.time, r)
        }
    }
    events.sort { $0.arrival < $1.arrival }
    var lastCamera: V3?
    var sightings: [(time: Double, id: Int, candidates: [PaddleTagSighting.Candidate], hidden: Bool)] = []
    var shown: [(t: Double, pose: PaddleState)] = []
    var result = Result()
    var prev: PaddleState?
    var next = 0
    var t = first + 0.5
    let end = events.last!.arrival
    while t < end {
        while next < events.count, events[next].arrival <= t {
            let r = events[next].record
            switch r.kind {
            case .imu(let s): fusion.add([s])
            case .tag(let id, let candidates, let camera):
                lastCamera = camera
                fusion.head = camera                    // the head, near enough (the camera is on it)
                if id == 67 {
                    table.add(candidates: candidates.map { ($0.rotation, $0.position) }, camera: camera, head: camera, time: r.time)
                } else {
                    let h = hidden(r.time)
                    sightings.append((r.time, id, candidates, h))
                    if !h { fusion.add(PaddleTagSighting(id: id, time: r.time, candidates: candidates)) }
                }
            case .palm(let p, let camera):
                if let camera { fusion.head = camera }
                fusion.addPalm(p, time: r.time, camera: camera ?? lastCamera)
            case .shown: break
            }
            next += 1
        }
        fusion.tick(time: t)
        if let s = table.shown(at: t), let a = table.anchor { result.tableSpread.append(vlength(s.centre - a.centre) * 1000) }
        if let p = fusion.pose(at: t + 0.03) {
            shown.append((t + 0.03, p))
            if let prev {
                let mm = vlength(p.center - prev.center) * 1000
                let expected = (vlength(p.velocity) + vlength(prev.velocity)) / 2 / 60 * 1000
                if mm > expected * 2 + 15 { result.jumps += 1 }
            }
            prev = p
        } else {
            result.missing += 1
            prev = nil
        }
        t += 1.0 / 60
    }
    result.seconds = end - first
    // The shown paddle at each sighting's time vs the paddle that tag alone gives (nearest candidate).
    let turns = fusion.calibration?.tagTurns ?? 0
    var i = 0
    for s in sightings {
        while i < shown.count, shown[i].t < s.time { i += 1 }
        guard i < shown.count, shown[i].t - s.time < 0.02 else { continue }
        let p = shown[i].pose, q = Quat(basis: p.basis)
        var best = (mm: Double.infinity, deg: Double.infinity)
        for c in s.candidates {
            let paddle: Quat, centre: V3
            if s.id == 0 {
                paddle = c.rotation * Quat(rotationVector: V3(0, 0, Double(turns) * .pi / 2))
                centre = c.position + paddle.act(PaddleFusion.tagToCentre)
            } else if let k = fusion.smallTurns[s.id], let offset = PaddleFusion.smallTags[s.id] {
                paddle = c.rotation * PaddleFusion.forehandPlacements[k]
                centre = c.position - paddle.act(offset)
            } else { continue }
            let deg = Quat.angle(paddle, q) * 180 / .pi
            if deg < best.deg { best = (vlength(centre - p.center) * 1000, deg) }
        }
        guard best.deg.isFinite else { continue }
        if s.hidden { result.hidden.append(best.mm) } else { result.visible.append(best.mm); result.visibleDeg.append(best.deg) }
    }
    return result
}

/// A recording with the drawn paddle (S lines): its records are in the order the app handled them, so it
/// replays exactly: each S is a frame, the paddle drawn at its camera image's moment.
func runAsApp(_ calibration: PaddleCalibration?, _ records: [Record], drop: (period: Double, length: Double, phase: Double)?) -> Result {
    let fusion = PaddleFusion()
    fusion.autoCalibrate = false
    fusion.calibration = calibration
    let first = records.first!.time
    func hidden(_ t: Double) -> Bool {
        guard let d = drop, t - first > 1 else { return false }
        return (t - first - 1 + d.phase).truncatingRemainder(dividingBy: d.period) < d.length
    }
    var lastCamera: V3?
    var sightings: [(time: Double, id: Int, candidates: [PaddleTagSighting.Candidate], hidden: Bool)] = []
    var shown: [Double: PaddleState] = [:]
    var result = Result()
    var prev: [V3] = []
    var prevPose: PaddleState?
    var prevTimes: [Double] = []
    var batch: [PaddleIMUSample] = []
    for r in records {
        if case .imu(let s) = r.kind { batch.append(s); continue }
        if !batch.isEmpty { fusion.add(batch); batch = [] }
        switch r.kind {
        case .imu: break
        case .tag(let id, let candidates, let camera):
            lastCamera = camera
            guard id != 67 else { break }
            let h = hidden(r.time)
            sightings.append((r.time, id, candidates, h))
            if !h { fusion.add(PaddleTagSighting(id: id, time: r.time, candidates: candidates)) }
        case .palm(let p, let camera):
            fusion.addPalm(p, time: r.time, camera: camera ?? lastCamera)
        case .shown(let app, let head):
            fusion.head = head
            fusion.tick(time: r.time)
            _ = fusion.pose(at: r.time + displayAhead)          // the physics asks first, for the display time
            guard let p = fusion.drawnPose(at: r.time) else { result.missing += 1; prev = []; prevTimes = []; prevPose = nil; continue }
            shown[(r.time * 1e4).rounded()] = p
            if drop == nil {
                result.vsApp.append(vlength(p.center - app) * 1000)
                if dump {
                    let o = fusion.offsetForDebug(at: r.time)
                    print(String(format: "D %.4f %.1f %.5f %.5f %.5f %.5f %.5f %.5f %@", r.time - first, vlength(p.center - app) * 1000,
                                 p.center.x, p.center.y, p.center.z, o.x, o.y, o.z, fusion.source))
                }
            }
            if let prevPose {
                let mm = vlength(p.center - prevPose.center) * 1000
                let expected = (vlength(p.velocity) + vlength(prevPose.velocity)) / 2 / 60 * 1000
                if mm > expected * 2 + 15 { result.jumps += 1 }
            }
            // A kink: the drawn paddle accelerating past 40 m/s^2 (the real one: 99% under 27).
            if let last = prevTimes.last, r.time - last > 0.008 {
                if prev.count == 2 {
                    let t0 = prevTimes[0], t1 = prevTimes[1], t2 = r.time
                    let a = ((p.center - prev[1]) / (t2 - t1) - (prev[1] - prev[0]) / (t1 - t0)) / ((t2 - t0) / 2)
                    if vlength(a) > 40 { result.kinks += 1 }
                }
                prev = (prev + [p.center]).suffix(2)
                prevTimes = (prevTimes + [r.time]).suffix(2)
            } else if prevTimes.isEmpty { prev = [p.center]; prevTimes = [r.time] }
            prevPose = p
        }
    }
    result.seconds = records.last!.time - first
    let turns = fusion.calibration?.tagTurns ?? 0
    for s in sightings {
        guard let p = shown[(s.time * 1e4).rounded()] else { continue }
        let q = Quat(basis: p.basis)
        var best = (mm: Double.infinity, deg: Double.infinity)
        for c in s.candidates {
            let paddle: Quat, centre: V3
            if s.id == 0 {
                paddle = c.rotation * Quat(rotationVector: V3(0, 0, Double(turns) * .pi / 2))
                centre = c.position + paddle.act(PaddleFusion.tagToCentre)
            } else if let k = fusion.smallTurns[s.id], let offset = PaddleFusion.smallTags[s.id] {
                paddle = c.rotation * PaddleFusion.forehandPlacements[k]
                centre = c.position - paddle.act(offset)
            } else { continue }
            let deg = Quat.angle(paddle, q) * 180 / .pi
            if deg < best.deg { best = (vlength(centre - p.center) * 1000, deg) }
        }
        guard best.deg.isFinite else { continue }
        if s.hidden { result.hidden.append(best.mm) } else { result.visible.append(best.mm); result.visibleDeg.append(best.deg) }
    }
    return result
}

/// How far ahead of the camera image the physics samples the paddle (the display time).
let dump = ProcessInfo.processInfo.environment["DUMP"] != nil
let displayAhead = Double(ProcessInfo.processInfo.environment["AHEAD"] ?? "0.045")!

func pct(_ a: [Double], _ q: Double) -> Double {
    guard !a.isEmpty else { return .nan }
    let s = a.sorted()
    return s[min(Int(Double(s.count - 1) * q), s.count - 1)]
}

if let text = ProcessInfo.processInfo.environment["PARAMS"] { FusionParams.current.set(text) }
let files = Array(CommandLine.arguments.dropFirst())
var all = Result()
for file in files {
    let (calibration, records) = load(file)
    for drop: (period: Double, length: Double, phase: Double)? in [nil, (2, 0.6, 0), (2, 0.6, 0.5), (2, 0.6, 1.0), (2, 0.6, 1.5)] {
        let asApp = records.contains { if case .shown = $0.kind { return true } else { return false } }
        let r = asApp ? runAsApp(calibration, records, drop: drop) : run(calibration, records, drop: drop)
        all.visible += r.visible
        all.hidden += r.hidden
        all.visibleDeg += r.visibleDeg
        all.jumps += r.jumps
        all.seconds += r.seconds
        all.missing += r.missing
        all.tableSpread += r.tableSpread
        all.kinks += r.kinks
        all.vsApp += r.vsApp
    }
}
let jumpsPer20 = Double(all.jumps) / all.seconds * 20
let kinksPer20 = Double(all.kinks) / all.seconds * 20
let score = pct(all.hidden, 0.5) + 0.5 * pct(all.hidden, 0.9) + pct(all.visible, 0.5) + 0.3 * pct(all.visible, 0.9) + 3 * jumpsPer20
    + 1 * kinksPer20
print(String(format: "SCORE %.1f | hidden p50 %.0f p90 %.0f | with tags p50 %.0f p90 %.0f mm, %.1f° | jumps %.1f /20 s | missing %d%@",
             score, pct(all.hidden, 0.5), pct(all.hidden, 0.9), pct(all.visible, 0.5), pct(all.visible, 0.9),
             pct(all.visibleDeg, 0.5), jumpsPer20, all.missing,
             all.tableSpread.isEmpty ? "" : String(format: " | table spread p50 %.1f mm", pct(all.tableSpread, 0.5)))
      + String(format: " | kinks %.1f /20 s", kinksPer20)
      + (all.vsApp.isEmpty ? "" : String(format: " | vs app p50 %.1f p90 %.1f mm", pct(all.vsApp, 0.5), pct(all.vsApp, 0.9))))
