import Foundation

/// Where the table goes, from its tag (67) lying on the real table: the middle of the player's half on
/// the tag, the surface at its height, the length along whichever of the tag's edges points most away
/// from the player.
///
/// One tag seen from a distance is noisy (centimetres, a few degrees), so:
/// - sightings are weighted by how close and how face-on the tag is, far ones (> 2.5 m) ignored;
/// - the first placement waits for several sightings that agree, and takes their weighted average;
/// - after that it refines slowly (time constant 3 s), keeping the same edge as the player walks around;
/// - it moves only when several sightings in a row agree on a spot more than 15 cm or 8° away (the tag
///   was moved), and the shown table glides there instead of jumping.
struct TableAnchor {
    struct Pose: Equatable {
        var centre: V3          // the tag's centre (the middle of the player's half), world
        var yaw: Double         // the table's heading (`PingPong.place`)
    }

    private(set) var anchor: Pose?
    private(set) var shown: Pose?
    private var anchorTime = 0.0
    private var shownTime = 0.0
    private var pending: [(pose: Pose, weight: Double)] = []

    static let farthest = 2.5
    static let settle = 3.0             // s
    static let glide = 0.15             // s
    static let moveDistance = 0.15, moveAngle = 8 * Double.pi / 180
    static let agreeing = 8

    mutating func reset() {
        anchor = nil
        shown = nil
        pending.removeAll()
    }

    /// A sighting: the tag's pose candidates (world <- tag, its centre), the camera's position, the
    /// player's head, its time.
    mutating func add(candidates: [(rotation: Quat, position: V3)], camera: V3, head: V3, time: Double) {
        // The pose that lies flat: a tag on a table faces up, so its z (into it) points down.
        guard let flat = candidates.max(by: { -$0.rotation.act(V3(0, 0, 1)).y < -$1.rotation.act(V3(0, 0, 1)).y })
        else { return }
        let down = -flat.rotation.act(V3(0, 0, 1)).y
        let flatLimit = cos(25 * Double.pi / 180)
        let distance = vlength(flat.position - camera)
        guard down > flatLimit, distance < Self.farthest else { return }
        let weight = clamped((Self.farthest - distance) / 1.0, 0.1, 1) * clamped((down - flatLimit) / (1 - flatLimit), 0.2, 1)

        let axes = [flat.rotation.act(V3(1, 0, 0)), flat.rotation.act(V3(0, 1, 0)),
                    -flat.rotation.act(V3(1, 0, 0)), -flat.rotation.act(V3(0, 1, 0))]
        let yaws = axes.map { atan2(-$0.x, -$0.z) }

        if let a = anchor {
            // Keep the same edge (it shouldn't swap as the player walks around).
            let yaw = yaws.min { abs(Self.wrap($0 - a.yaw)) < abs(Self.wrap($1 - a.yaw)) }!
            let pose = Pose(centre: flat.position, yaw: yaw)
            if vlength(pose.centre - a.centre) < Self.moveDistance, abs(Self.wrap(yaw - a.yaw)) < Self.moveAngle {
                pending.removeAll()
                let k = (1 - exp(-min(max(time - anchorTime, 0), 0.5) / Self.settle)) * weight
                anchor = Pose(centre: a.centre + (pose.centre - a.centre) * k, yaw: a.yaw + Self.wrap(yaw - a.yaw) * k)
                anchorTime = time
            } else {
                gather(pose, weight: weight, time: time)
            }
        } else {
            // First placement: the edge pointing most away from the player.
            let away = V3(flat.position.x - head.x, 0, flat.position.z - head.z)
            let best = axes.indices.max { vdot(V3(axes[$0].x, 0, axes[$0].z), away) < vdot(V3(axes[$1].x, 0, axes[$1].z), away) }!
            gather(Pose(centre: flat.position, yaw: yaws[best]), weight: weight, time: time)
        }
    }

    /// Sightings away from the anchor (or before there is one): once enough agree, they become it.
    private mutating func gather(_ pose: Pose, weight: Double, time: Double) {
        if let first = pending.first?.pose,
           vlength(pose.centre - first.centre) > 0.08 || abs(Self.wrap(pose.yaw - first.yaw)) > 10 * Double.pi / 180 {
            pending.removeAll()             // they don't agree: start over from this one
        }
        pending.append((pose, weight))
        guard pending.count >= Self.agreeing else { return }
        let total = pending.reduce(0) { $0 + $1.weight }
        let centre = pending.reduce(V3.zero) { $0 + $1.pose.centre * $1.weight } / total
        let s = pending.reduce(0) { $0 + sin($1.pose.yaw) * $1.weight }, c = pending.reduce(0) { $0 + cos($1.pose.yaw) * $1.weight }
        anchor = Pose(centre: centre, yaw: atan2(s, c))
        anchorTime = time
        pending.removeAll()
    }

    /// The table as shown at `time`: gliding toward the anchor (a moved tag doesn't make it jump).
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
