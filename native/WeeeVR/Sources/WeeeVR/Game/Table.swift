/// Regulation table tennis table (ITTF) in its own frame, which the whole game uses: origin at the
/// centre of the playing surface, y up, z along the table toward the player (the player's end line
/// is at z = +length/2, the opponent's at -length/2), x across. The shader mirrors these numbers.
enum Table {
    static let length = 2.74
    static let width = 1.525
    static let height = 0.76                // playing surface above the floor
    static let thickness = 0.025
    static let netHeight = 0.1525
    static let netOverhang = 0.1525         // net beyond each side line
    static let floorY = -height

    static var halfLength: Double { length / 2 }
    static var halfWidth: Double { width / 2 }
    static var netHalfSpan: Double { halfWidth + netOverhang }

    /// Where a new ball rests: on the player's half, to their left (picked up with the left hand).
    static let spawn = V3(-0.35, 0.02, halfLength - 0.32)
}
