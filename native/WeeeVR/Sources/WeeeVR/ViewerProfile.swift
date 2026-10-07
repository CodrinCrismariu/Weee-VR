import Foundation

/// Physical description of the Cardboard viewer + phone screen, plus user toggles.
/// Defaults are Google Cardboard v2 ("Cardboard I/O 2015") on an iPhone 15.
struct ViewerProfile: Codable, Equatable {
    // Viewer optics
    var interLensMM: Float = 63.9
    var screenToLensMM: Float = 39.0
    var trayToLensMM: Float = 35.0
    var k1: Float = 0.34
    var k2: Float = 0.55
    var lensFovDeg: Float = 50          // half-angle visible through each lens

    // Phone
    var screenPPI: Float = 460          // iPhone 15
    var bottomBezelMM: Float = 3.5      // phone edge resting on the tray -> active screen area (approx.)

    // Rendering
    var fovScale: Float = 1.0           // 1 = life-size; >1 squeezes more camera FOV into the lens
    var ipdMM: Float = 63.0             // eye separation for both the passthrough reprojection and the overlay
    var flipCamera = false              // rotate camera image 180° if the phone sits the other way round
    var predictionMS: Float = 4         // extra pose prediction past the target present time (scan-out)

    // Camera vs eyes: midpoint between the eyes relative to the passthrough camera, in the landscape
    // view frame (x right, y up, z toward the viewer). iPhone 15 main camera in a Cardboard: the camera
    // is ~48 mm left of, ~11 mm above and ~62 mm in front of the eyes.
    var eyeOffsetXMM: Float = 48
    var eyeOffsetYMM: Float = -11
    var eyeOffsetZMM: Float = 62

    // Depth proxy: the camera image is projected onto a sphere this far from the eyes, then viewed
    // from each eye's real position. Auto follows ARKit's feature points.
    var autoDepth = true
    var fixedDepthM: Float = 1.5

    // Tracking + passthrough camera. ARKit only runs on the 1×; XRSLAM gives 6DoF on the 0.5×.
    var passthrough = Passthrough.xrslam
    var lateLatch = true                // render just before the display deadline with the newest frame

    // XRSLAM camera-IMU calibration and tracker size (changing any of these restarts the tracker).
    var slamTimeOffsetMS: Float = 0     // added to camera timestamps to line them up with the IMU
    var slamLeverXMM: Float = 0         // ultra-wide position relative to the IMU, device frame
    var slamLeverYMM: Float = 0
    var slamLeverZMM: Float = 0
    var slamFeedHz: Float = 30          // camera frames per second given to the tracker
    var slamKeypoints: Float = 300
    var slamWindow: Float = 10          // sliding-window keyframes (XRSLAM's benchmark setting; its iOS app used 5)
    var slamDefaultsVersion = 2         // bump to reset saved slam* values to new defaults
    var defaultsVersion = 4             // bump (and list keys in `resetOnUpgrade`) to push new defaults

    // Hands (Vision hand pose -> 3D hands)
    var handTracking = true
    var handOverlay = true              // draw the tracked 3D hands
    var handScale: Float = 1            // hand size relative to an average adult (sets bone thickness)

    // Toggles
    var reprojection = true             // gyro "timewarp" of the passthrough to display time
    var showGrid = false                // world-locked 15° grid + horizon (debug)
    var showCube = true                 // world-locked wireframe cube 1 m ahead

    private static let storageKey = "viewerProfile.v2"

    /// Saved values replaced by the current defaults when a profile older than the version is loaded.
    private static let resetOnUpgrade: [Int: [String]] = [
        3: ["passthrough", "showGrid"],     // XRSLAM 0.5× by default, grid off
        4: ["passthrough"],                 // back to XRSLAM after swipe-switching was removed
    ]

    /// Saved values overlaid on the defaults, so fields added since the last save keep their default
    /// and removed ones are ignored instead of discarding the whole calibration.
    static func load() -> ViewerProfile {
        guard let data = UserDefaults.standard.data(forKey: storageKey),
              let saved = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let defaultsData = try? JSONEncoder().encode(ViewerProfile()),
              let defaults = try? JSONSerialization.jsonObject(with: defaultsData) as? [String: Any]
        else { return ViewerProfile() }
        var merged = defaults
        let keepSlam = (saved["slamDefaultsVersion"] as? Int ?? 0) >= ViewerProfile().slamDefaultsVersion
        let savedVersion = saved["defaultsVersion"] as? Int ?? 0
        let reset = resetOnUpgrade.filter { $0.key > savedVersion }.flatMap(\.value)
        for (key, value) in saved where merged[key] != nil && (keepSlam || !key.hasPrefix("slam"))
            && !reset.contains(key) {
            merged[key] = value
        }
        guard let mergedData = try? JSONSerialization.data(withJSONObject: merged),
              let profile = try? JSONDecoder().decode(ViewerProfile.self, from: mergedData)
        else { return ViewerProfile() }
        return profile
    }

    func save() {
        if let data = try? JSONEncoder().encode(self) {
            UserDefaults.standard.set(data, forKey: Self.storageKey)
        }
    }

    var pixelsPerMeter: Float { screenPPI / 0.0254 }

    var eyeOffset: SIMD3<Float> { SIMD3(eyeOffsetXMM, eyeOffsetYMM, eyeOffsetZMM) / 1000 }

    var slamSettings: XRSLAMSource.Settings {
        XRSLAMSource.Settings(
            timeOffsetMS: slamTimeOffsetMS,
            leverArmMM: SIMD3(slamLeverXMM, slamLeverYMM, slamLeverZMM),
            feedHz: Int(slamFeedHz.rounded()),
            keypoints: Int(slamKeypoints.rounded()),
            window: Int(slamWindow.rounded())
        )
    }

    /// Lens centres in drawable pixels (origin top-left), for a landscape drawable of `size`.
    func lensCenters(drawableWidth w: Float, height h: Float) -> (left: SIMD2<Float>, right: SIMD2<Float>) {
        let halfSeparation = interLensMM / 1000 / 2 * pixelsPerMeter
        let y = h - (trayToLensMM - bottomBezelMM) / 1000 * pixelsPerMeter
        return (SIMD2(w / 2 - halfSeparation, y), SIMD2(w / 2 + halfSeparation, y))
    }
}

enum Passthrough: String, Codable, CaseIterable {
    case xrslam         // XRSLAM 6DoF, 0.5× camera
    case arkit          // ARKit 6DoF, 1× camera
    case ultraWide      // gyro 3DoF, 0.5× camera
    case main           // gyro 3DoF, 1× camera

    var label: String {
        switch self {
        case .xrslam: "XRSLAM 6DoF · 0.5×"
        case .arkit: "ARKit 6DoF · 1×"
        case .ultraWide: "Gyro 3DoF · 0.5×"
        case .main: "Gyro 3DoF · 1×"
        }
    }
}
