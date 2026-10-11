import simd

/// In-headset control (the screen can't be tapped inside a closed headset): the open left palm turned
/// toward the face and held for `holdTime` recenters. `progress` (0...1) lights up the left hand
/// meanwhile. After it fires, the hand has to drop the pose before it can fire again.
///
/// Open palm: at least three fingers extended (fingertip much farther from the wrist than the
/// knuckle). Facing: the palm's normal points at the eyes. Works with the joints in either placement
/// (overlay or true depth): it only uses directions and ratios. Render thread.
final class PalmGesture {
    static let holdTime = 1.0
    private static let dropout = 0.15           // s the pose may flicker out without restarting

    private(set) var progress: Float = 0
    private var since: Double?
    private var lastSeen = -1.0
    private var armed = true

    /// `joints`: the left hand (world, MediaPipe order); `head`: the eyes' midpoint. True once, when
    /// the gesture completes.
    func update(time: Double, joints: [SIMD3<Float>]?, head: SIMD3<Float>) -> Bool {
        guard let joints, Self.isOpenPalmFacing(joints, head: head) else {
            if time - lastSeen > Self.dropout {
                since = nil
                progress = 0
                armed = true
            }
            return false
        }
        lastSeen = time
        guard armed else { return false }
        let start = since ?? time
        since = start
        progress = Float(min((time - start) / Self.holdTime, 1))
        guard progress >= 1 else { return false }
        armed = false
        since = nil
        progress = 0
        return true
    }

    private static func isOpenPalmFacing(_ j: [SIMD3<Float>], head: SIMD3<Float>) -> Bool {
        guard j.count >= 21 else { return false }
        let wrist = j[0]
        let knuckles = (j[5] + j[9] + j[13] + j[17]) / 4
        let forward = simd_normalize(knuckles - wrist)
        var across = j[5] - j[17]                                   // toward the index side
        across = simd_normalize(across - forward * simd_dot(across, forward))
        let palmNormal = simd_cross(forward, across)                // out of a left palm
        let palm = (j[0] + j[5] + j[9] + j[13] + j[17]) / 5
        guard simd_dot(palmNormal, simd_normalize(head - palm)) > 0.7 else { return false }
        var extended = 0
        for (knuckle, tip) in [(5, 8), (9, 12), (13, 16), (17, 20)]
        where simd_distance(j[tip], wrist) > 1.6 * simd_distance(j[knuckle], wrist) {
            extended += 1
        }
        return extended >= 3
    }
}
