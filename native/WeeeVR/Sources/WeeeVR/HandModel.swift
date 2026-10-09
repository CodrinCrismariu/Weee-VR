import Foundation
import simd

/// Render-thread side of hand tracking: Vision's 2D joints -> world-space 3D joints, filtered and
/// predicted, packed for the shader (`HandModel.stride` float4 per hand, see ShaderSource).
///
/// - Placement: each joint goes where the passthrough shows it, i.e. where its camera ray meets the
///   passthrough's depth-proxy sphere (radius D around the eyes). The 3D hand then covers the
///   passthrough hand in both eyes by construction. (Placing it at the hand's true depth would not:
///   the passthrough draws the hand at depth D, a few degrees away from where it really is.)
/// - Size: bones are scaled by D / (true hand distance) so they keep a finger's real angular width.
///   The true distance comes from palm proportions (wrist to knuckles ~8 cm, scaled by `handScale`).
/// - World: joints are placed with the camera pose at the frame's capture time, so head motion is
///   fully compensated; only the hand's own motion has to be predicted.
/// - Filter: One Euro (adaptive low-pass: smooth when still, responsive when moving) per joint in
///   world space, then extrapolated with its velocity to the capture time of the passthrough frame
///   being drawn: the 3D hand has to sit on the hand you see, which is the camera's (~50 ms old)
///   picture, not on where the hand physically is at display time.
final class HandModel {
    static let stride = 23              // float4 per hand: info, bounding sphere, 21 joints
    static let empty = [SIMD4<Float>](repeating: .zero, count: 2 * stride)
    static let maxPrediction = 0.06     // s, passthrough overlay
    static let timeout = 0.2            // s without a detection hides the hand
    /// s a hand the tracker lost stays (frozen) before it hides. The game holds it longer: a paddle
    /// mustn't blink out during a blurred swing.
    var lostHold = 0.12

    /// Where joints are placed. Overlay: on the passthrough's depth sphere, covering the camera's hand
    /// in both eyes. True depth: at the hand's real distance (from palm proportions), real size, for
    /// interacting with virtual objects at their real distances (the game).
    var trueDepth = false {
        didSet { if trueDepth != oldValue { reset() } }
    }

    /// Palm segments with typical adult lengths (m): wrist-index MCP, wrist-middle MCP, wrist-little
    /// MCP, index MCP-little MCP. Foreshortening only makes a segment look shorter (the hand look
    /// farther), so the nearest estimate wins.
    static let palmSegments: [(Int, Int, Float)] = [(0, 5, 0.080), (0, 9, 0.083), (0, 17, 0.070), (5, 17, 0.058)]

    private struct Slot {
        var id: Int
        var chirality: Int
        var filters = [OneEuro3](repeating: OneEuro3(), count: HandTracker.jointCount)
        var valid = [Bool](repeating: false, count: HandTracker.jointCount)
        var time = 0.0
        var lost = false                // a newer detection didn't have it: hold still, then hide
        var lead: Float = 0             // prediction used while it was tracked
        var radiusScale: Float = 1
    }

    private var slots: [Slot?] = [nil, nil]
    private var lastSequence: UInt64 = 0

    /// Integrate a detection if it is new. `pose` is world <- camera (view frame) at its capture time;
    /// `head` and `proxyDepth` describe the passthrough's depth sphere.
    func update(_ detection: HandTracker.Detection?, pose: (simd_float3x3, SIMD3<Float>)?, head: SIMD3<Float>,
                proxyDepth: Float, flip: Bool, handScale: Float) {
        guard let detection, detection.sequence != lastSequence else { return }
        lastSequence = detection.sequence
        guard let (rotation, origin) = pose else { return }
        let k = detection.intrinsics
        let relative = origin - head
        var taken = [false, false]

        for hand in detection.hands {
            var rays = [SIMD3<Float>](repeating: .zero, count: HandTracker.jointCount)
            var directions = [SIMD3<Float>](repeating: .zero, count: HandTracker.jointCount)
            var sphereT = [Float](repeating: 0, count: HandTracker.jointCount)
            var sphereRange: Float = 0
            for i in 0..<HandTracker.jointCount where hand.valid[i] {
                var px = hand.points[i]
                if flip { px = detection.size - px }
                // Camera ray (view frame: x right, y up, -z forward), then where it meets the sphere.
                rays[i] = simd_normalize(SIMD3((px.x - k.z) / k.x, (k.w - px.y) / k.y, -1))
                directions[i] = rotation * rays[i]
                let b = simd_dot(directions[i], relative)
                let c = simd_length_squared(relative) - proxyDepth * proxyDepth
                sphereT[i] = -b + sqrt(max(b * b - c, 0))
                sphereRange += sphereT[i]
            }
            sphereRange /= Float(max(hand.valid.filter { $0 }.count, 1))

            var trueRange: Float = .infinity
            for (a, b, length) in Self.palmSegments where hand.valid[a] && hand.valid[b] {
                let angle = acos(min(max(simd_dot(rays[a], rays[b]), -1), 1))
                if angle > 1e-4 { trueRange = min(trueRange, length * handScale / (2 * sin(angle / 2))) }
            }
            trueRange = trueRange.isFinite ? min(max(trueRange, 0.15), 1.2) : sphereRange
            let radiusScale = trueDepth ? 1 : sphereRange / trueRange

            // MediaPipe's relative depth (image pixels, wrist-relative) -> metres at the hand. Overlay:
            // scaled like the rest of the hand onto the sphere, along each joint's own camera ray, so
            // the hand still covers the passthrough hand. True depth: from the hand's real distance.
            let metresPerPixel = trueRange / k.x * radiusScale
            var world = [SIMD3<Float>](repeating: .zero, count: HandTracker.jointCount)
            for i in 0..<HandTracker.jointCount where hand.valid[i] {
                let base = trueDepth ? trueRange : sphereT[i]
                let t = base + (hand.depths[i] - hand.depths[0]) * metresPerPixel
                world[i] = origin + directions[i] * t
            }

            let index = slotIndex(for: hand.id, taken: taken)
            taken[index] = true
            var slot = slots[index] ?? Slot(id: hand.id, chirality: hand.chirality)
            let dt = Float(detection.captureTime - slot.time)
            if slots[index] == nil || slot.id != hand.id || dt > Float(Self.timeout) || dt <= 0 {
                slot = Slot(id: hand.id, chirality: hand.chirality)
                slot.radiusScale = radiusScale
            }
            for i in 0..<HandTracker.jointCount where hand.valid[i] {
                _ = slot.filters[i].filter(world[i], dt: dt)
            }
            slot.radiusScale += (radiusScale - slot.radiusScale) * 0.2
            slot.valid = hand.valid
            slot.time = detection.captureTime
            slot.chirality = hand.chirality
            slot.lost = false
            slots[index] = slot
        }
        // Not in this detection: lost. Extrapolating it would fling it off along its last velocity
        // (straight ahead in a punch), so it is held where it was and hidden soon after.
        for index in 0..<2 where !taken[index] { slots[index]?.lost = true }
    }

    func reset() { slots = [nil, nil] }

    var visibleCount: Int { slots.compactMap { $0 }.count }

    /// Shader data for both hands, predicted to `time` (overlay: the passthrough frame's capture time;
    /// game: the display time), at most `maxLead` past their detection.
    func uniforms(time displayTime: Double, maxLead: Double = HandModel.maxPrediction) -> [SIMD4<Float>] {
        var out = Self.empty
        for h in 0..<2 {
            guard let slot = slots[h], displayTime - slot.time < (slot.lost ? lostHold : Self.timeout) else {
                slots[h] = nil
                continue
            }
            let base = h * Self.stride
            let lead = slot.lost ? slot.lead : Float(min(max(displayTime - slot.time, 0), maxLead))
            slots[h]?.lead = lead
            var joints = [SIMD3<Float>](repeating: .zero, count: HandTracker.jointCount)
            var centre = SIMD3<Float>.zero
            var count: Float = 0
            for i in 0..<HandTracker.jointCount where slot.valid[i] {
                joints[i] = slot.filters[i].predict(seconds: lead)
                centre += joints[i]
                count += 1
            }
            guard count > 0 else { continue }
            centre /= count
            var radius: Float = 0
            for i in 0..<HandTracker.jointCount where slot.valid[i] {
                out[base + 2 + i] = SIMD4(joints[i], 1)
                radius = max(radius, simd_length(joints[i] - centre))
            }
            out[base + 1] = SIMD4(centre, radius + 0.02 * slot.radiusScale)
            out[base] = SIMD4(1, slot.radiusScale, 0, Float(slot.chirality))
        }
        return out
    }

    /// Joints (world, MediaPipe order) of the shown hand with `chirality` (0 left, 1 right), as
    /// `uniforms` last predicted them; nil when that hand isn't shown. Call after `uniforms`.
    func joints(chirality: Int) -> [SIMD3<Float>]? {
        guard let slot = slots.first(where: { $0?.chirality == chirality }) ?? nil,
              slot.valid.allSatisfy({ $0 }) else { return nil }
        return slot.filters.map { $0.predict(seconds: slot.lead) }
    }

    /// The tracker keeps each hand's id across frames: same id -> same slot (keeps its filter), else
    /// an empty slot, else the one updated longest ago.
    private func slotIndex(for id: Int, taken: [Bool]) -> Int {
        if let same = (0..<2).first(where: { !taken[$0] && slots[$0]?.id == id }) { return same }
        if let empty = (0..<2).first(where: { !taken[$0] && slots[$0] == nil }) { return empty }
        return (0..<2).filter { !taken[$0] }.min { (slots[$0]?.time ?? 0) < (slots[$1]?.time ?? 0) } ?? 0
    }
}

/// One Euro filter (Casiez et al. 2012) on a 3D point, also tracking its filtered velocity.
struct OneEuro3 {
    static let minCutoff: Float = 3.0       // Hz when still
    static let beta: Float = 6.0            // cutoff increase per m/s
    static let derivativeCutoff: Float = 5.0

    private(set) var value: SIMD3<Float>?
    private(set) var velocity = SIMD3<Float>.zero

    mutating func filter(_ x: SIMD3<Float>, dt: Float) -> SIMD3<Float> {
        guard let previous = value, dt > 0 else {
            value = x
            velocity = .zero
            return x
        }
        let rawVelocity = (x - previous) / dt
        velocity += Self.alpha(Self.derivativeCutoff, dt) * (rawVelocity - velocity)
        let cutoff = Self.minCutoff + Self.beta * simd_length(velocity)
        let filtered = previous + Self.alpha(cutoff, dt) * (x - previous)
        value = filtered
        return filtered
    }

    func predict(seconds: Float) -> SIMD3<Float> { (value ?? .zero) + velocity * seconds }

    private static func alpha(_ cutoff: Float, _ dt: Float) -> Float {
        let tau = 1 / (2 * Float.pi * cutoff)
        return 1 / (1 + tau / dt)
    }
}
