import Foundation

/// Where the table goes, from its tag (67) lying on the real table: the middle of the player's half on
/// the tag, the surface at its height, and the direction from how the tag lies (`labelSide`: where its
/// printed ID label, under the image, faces). Never from where the player happens to stand when the tag
/// is first seen, so it is the same every time.
///
/// The tag's frame is AprilTag's: x right, y down the printed image (toward the label), z into the tag;
/// the paddle's calibration confirms it (its tag's +y points down the print, toward the handle).
///
/// One tag seen from 1-2 m is noisy (a centimetre or two, a degree or two), and a table that follows
/// that noise wobbles, worst at its far end (1° is 4 cm there). So:
/// - sightings are weighted by distance (full up to 1.2 m, none past 2.5 m) and by how face-on the tag
///   is seen (a grazing view's heading is poor);
/// - the first placement waits for `placeCount` good sightings over at least 0.6 s and takes the
///   median of those near the median;
/// - after that the table holds still: a running median of the last 1.5 s within `holdDistance` /
///   `holdAngle` of it changes nothing; beyond that it is pulled over slowly (`refine`); past
///   `moveDistance` / `moveAngle` for longer than `moveAfter` (the tag was moved) it goes there;
/// - the shown table glides to the anchor (`glide`), so no change ever jumps.
struct TableAnchor {
    struct Pose: Equatable {
        var centre: V3          // the tag's centre (the middle of the player's half), world
        var yaw: Double         // the table's heading (`PingPong.place`)
    }

    /// Where the tag's printed label faces, seen from the player's end: 0 the player, 1 the far end,
    /// 2 the player's left, 3 the player's right.
    var labelSide = 0

    /// The far end's direction in the tag's frame, for each `labelSide`.
    static let forwardInTag = [V3(0, -1, 0), V3(0, 1, 0), V3(-1, 0, 0), V3(1, 0, 0)]

    private(set) var anchor: Pose?
    private(set) var shown: Pose?
    private var shownTime = 0.0
    private var lastTime = 0.0
    private var placing: [(pose: Pose, weight: Double, time: Double)] = []
    private var recent: [(pose: Pose, weight: Double, time: Double)] = []
    private var movingSince: Double?

    static let farthest = 2.5, sharpest = 1.2         // m: weight 0 beyond, 1 within
    static let placeCount = 12, placeSpan = 0.6        // sightings, s
    static let window = 1.5                            // s of recent sightings
    static let holdDistance = 0.03, holdAngle = 2 * Double.pi / 180
    static let refine = 5.0                            // s: time constant of small corrections
    static let moveDistance = 0.15, moveAngle = 8 * Double.pi / 180, moveAfter = 1.0
    static let glide = 0.5                             // s

    mutating func reset() {
        anchor = nil
        shown = nil
        placing.removeAll()
        recent.removeAll()
        movingSince = nil
    }

    /// A sighting: the tag's pose candidates (world <- tag, its centre), the camera's position, the
    /// player's head, its time.
    mutating func add(candidates: [(rotation: Quat, position: V3)], camera: V3, head: V3, time: Double) {
        // The pose that lies flat: a tag on a table faces up, so its z (into it) points down.
        guard let flat = candidates.max(by: { -$0.rotation.act(V3(0, 0, 1)).y < -$1.rotation.act(V3(0, 0, 1)).y })
        else { return }
        let down = -flat.rotation.act(V3(0, 0, 1)).y
        let distance = vlength(flat.position - camera)
        guard down > cos(25 * Double.pi / 180), distance < Self.farthest else { return }
        // How face-on: the camera's ray against the tag's up (1 straight down onto it).
        let faceOn = vdot(vnormalize(camera - flat.position), V3(0, 1, 0))
        let weight = clamped((Self.farthest - distance) / (Self.farthest - Self.sharpest), 0, 1)
            * clamped((faceOn - 0.25) / 0.45, 0.1, 1)
        guard weight > 0.05 else { return }

        // The far end's direction, from how the tag lies.
        let axis = flat.rotation.act(Self.forwardInTag[min(max(labelSide, 0), 3)])
        let forward = vnormalize(V3(axis.x, 0, axis.z))
        let pose = Pose(centre: flat.position, yaw: atan2(-forward.x, -forward.z))
        let dt = min(max(time - lastTime, 0), 0.5)
        lastTime = time

        guard let a = anchor else {
            place(pose, weight: weight, time: time)
            return
        }
        recent.append((pose, weight, time))
        recent.removeAll { time - $0.time > Self.window }
        guard recent.count >= 6, let estimate = Self.robust(recent.map { ($0.pose, $0.weight) }) else { return }
        let offset = vlength(estimate.centre - a.centre), turn = abs(Self.wrap(estimate.yaw - a.yaw))
        if offset > Self.moveDistance || turn > Self.moveAngle {
            // Far off: the tag was moved, if it stays there.
            if movingSince == nil { movingSince = time }
            if time - movingSince! > Self.moveAfter {
                anchor = estimate
                recent.removeAll()
                movingSince = nil
            }
            return
        }
        movingSince = nil
        guard offset > Self.holdDistance || turn > Self.holdAngle else { return }        // noise: hold still
        let k = 1 - exp(-dt / Self.refine)
        anchor = Pose(centre: a.centre + (estimate.centre - a.centre) * k, yaw: a.yaw + Self.wrap(estimate.yaw - a.yaw) * k)
    }

    /// Before the first placement: gather sightings that agree; place on their robust average.
    private mutating func place(_ pose: Pose, weight: Double, time: Double) {
        if let first = placing.first?.pose,
           vlength(pose.centre - first.centre) > 0.1 || abs(Self.wrap(pose.yaw - first.yaw)) > 15 * Double.pi / 180 {
            placing.removeAll()             // they don't agree: start over from this one
        }
        placing.append((pose, weight, time))
        guard placing.count >= Self.placeCount, time - placing[0].time >= Self.placeSpan,
              let estimate = Self.robust(placing.map { ($0.pose, $0.weight) }) else { return }
        anchor = estimate
        placing.removeAll()
    }

    /// A robust average: the median, then the weighted mean of the sightings within 3 cm and 3° of it.
    static func robust(_ items: [(pose: Pose, weight: Double)]) -> Pose? {
        guard !items.isEmpty else { return nil }
        func median(_ v: [Double]) -> Double { let s = v.sorted(); return s[s.count / 2] }
        let mx = median(items.map { $0.pose.centre.x }), my = median(items.map { $0.pose.centre.y })
        let mz = median(items.map { $0.pose.centre.z })
        let reference = items[0].pose.yaw
        let myaw = reference + median(items.map { wrap($0.pose.yaw - reference) })
        let m = V3(mx, my, mz)
        let near = items.filter { vlength($0.pose.centre - m) < 0.03 && abs(wrap($0.pose.yaw - myaw)) < 3 * Double.pi / 180 }
        let use = near.count >= max(3, items.count / 3) ? near : items
        let total = use.reduce(0) { $0 + $1.weight }
        guard total > 0 else { return Pose(centre: m, yaw: myaw) }
        let centre = use.reduce(V3.zero) { $0 + $1.pose.centre * $1.weight } / total
        let yaw = myaw + use.reduce(0) { $0 + wrap($1.pose.yaw - myaw) * $1.weight } / total
        return Pose(centre: centre, yaw: wrap(yaw))
    }

    /// The table as shown at `time`: gliding toward the anchor, so no change makes it jump.
    mutating func shown(at time: Double) -> Pose? {
        guard let a = anchor else { return nil }
        guard let s = shown, time > shownTime else {
            if shown == nil { shown = a; shownTime = time }
            return shown
        }
        let k = 1 - exp(-min(time - shownTime, 0.1) / Self.glide)
        shown = Pose(centre: s.centre + (a.centre - s.centre) * k, yaw: s.yaw + Self.wrap(a.yaw - s.yaw) * k)
        shownTime = time
        return shown
    }

    static func wrap(_ a: Double) -> Double { atan2(sin(a), cos(a)) }
}
