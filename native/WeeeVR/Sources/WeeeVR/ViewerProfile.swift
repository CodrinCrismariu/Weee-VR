import Foundation

/// Physical description of the phone viewer (headset) + phone screen, plus user toggles.
/// Defaults are the VRG Pro (`Headset.vrgPro`) on an iPhone 15.
struct ViewerProfile: Codable, Equatable {
    // Viewer optics (`Headset` presets; the sliders fine-tune them)
    var headset = Headset.vrgPro
    var interLensMM: Float = 63.0
    var screenToLensMM: Float = 42.0
    var trayToLensMM: Float = 36.0
    var k1: Float = 0.38
    var k2: Float = 0.40
    var lensFovDeg: Float = 55          // half-angle visible through each lens

    // Phone
    var screenPPI: Float = 460          // iPhone 15
    var bottomBezelMM: Float = 3.5      // phone edge resting on the tray -> active screen area (approx.)

    // Rendering
    var fovScale: Float = 1.0           // 1 = life-size; >1 squeezes more camera FOV into the lens
    var ipdMM: Float = 63.0             // eye separation for both the passthrough reprojection and the overlay
    var flipCamera = false              // rotate camera image 180° if the phone sits the other way round
    var predictionMS: Float = 4         // extra pose prediction past the target present time (scan-out)

    // Camera vs eyes: midpoint between the eyes relative to the iPhone 15's 1× camera, in the landscape
    // view frame (x right, y up, z toward the viewer). In a Cardboard the camera is ~48 mm left of,
    // ~11 mm above and ~62 mm in front of the eyes; the VRG Pro holds the phone ~10 mm farther out.
    var eyeOffsetXMM: Float = 48
    var eyeOffsetYMM: Float = -11
    var eyeOffsetZMM: Float = 72

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
    var defaultsVersion = 9             // bump (and list keys in `resetOnUpgrade`) to push new defaults

    // Hands (Vision hand pose -> 3D hands)
    var handTracking = true
    var handOverlay = true              // draw the tracked 3D hands
    var handScale: Float = 1            // hand size relative to an average adult (sets bone thickness)

    // Image (all in the passthrough shader; none of it reaches the trackers)
    var accurateColor = true            // camera's own YCbCr matrix (BT.709) + sRGB -> P3 for the screen
    var chromaticAberration: Float = 0  // lens colour fringes: red sampled at 1 - c, blue at 1 + c
    var sharpening: Float = 0.35        // bicubic + adaptive unsharp on the camera image (0 = bilinear)
    var blackLift: Float = 0.02         // keeps OLED pixels off pure black: less smear in dark areas
    var edgeSoftness: Float = 0.12      // fraction of the lens radius fading to black at its edge
    var maxBrightness = true            // screen at full brightness while the VR view is up

    // Toggles
    var reprojection = true            // gyro "timewarp" of the passthrough to display time
    var showGrid = false                // world-locked 15° grid + horizon (debug)

    // Game
    var scene = GameScene.pingPong      // what is placed ahead on a tap
    var tableTag = true                 // put the table on its AprilTag (ID 67, 102.2 mm) when it's seen
    var mqttEnabled = true              // post the record of continuous hits over MQTT
    var mqttHost = "test.mosquitto.org" // broker (plain TCP, port 1883, no login)
    var mqttTopic = "ME193/Rogers/Codrin"
    var tableTagLabel = 0               // where tag 67's printed label faces: 0 me, 1 the far end, 2 my left, 3 my right
    var tableDropM: Float = 0.80        // table surface below the eyes (standing ~0.8, seated ~0.45)
    var opponentLevel: Float = 0.5      // 0 gentle ... 1 fast
    var assist: Float = 0.5             // 0 realistic ... 1: forgiving paddle, steered shots, slower ball
    var floatingServe = true            // a new ball hovers at serve height, ready to hit
    var gripAngleDeg: Float = 40        // paddle blade from the knuckle direction toward the thumb
    var gripOffsetMM: Float = -12       // blade plane along the palm normal (- = into the fist)
    var paddlePrediction: Float = 0     // s the 3D hands lead the camera image (0: exactly on it)
    var gameVolume: Float = 0.8
    var maxExposureMS: Float = 3        // camera exposure cap (auto below it): less motion blur (the paddle's tags stay readable in swings), more grain; 17 = off

    // Smart paddle (ArduinoApps/smart-paddle): IMU + AprilTag paddle with haptics, over Wi-Fi
    var smartPaddle = true              // play with it whenever it's connected and calibrated
    var paddleHost = "10.243.94.25"     // the board's address
    var paddleBladeWidthMM: Float = 150 // the real paddle's size (the drawn one and its hit area match it)
    var paddleBladeLengthMM: Float = 158
    var paddleHandleMM: Float = 100
    var paddleOnCameraImage = true      // the paddle drawn at the camera image's moment (on the real one)
    var paddleHaptics: Float = 1        // vibration strength on a hit (0 = off); harder hits buzz more
    var paddleTagTurns = -1             // the tag's quarter turns on the blade; -1 = as calibrated
    // Its calibration ("Calibrate paddle", once): IMU <- tag rotation (w x y z; empty = none), clock
    // offset, tag turns and fit residual.
    var paddleMount: [Double] = []
    var paddleClockOffsetMS: Double = 0
    var paddleCalTurns = 0
    var paddleCalResidualDeg: Double = 0
    var paddleSmallTags: [Int] = []     // learned: ID, quarter turns, ID, quarter turns (forehand tags 68, 69)

    var paddleSmallTagTurns: [Int: Int] {
        get {
            var out: [Int: Int] = [:]
            for i in Swift.stride(from: 0, to: paddleSmallTags.count - 1, by: 2) { out[paddleSmallTags[i]] = paddleSmallTags[i + 1] }
            return out
        }
        set { paddleSmallTags = newValue.sorted { $0.key < $1.key }.flatMap { [$0.key, $0.value] } }
    }

    var paddleCalibration: PaddleCalibration? {
        get {
            paddleMount.count == 4 ? PaddleCalibration(mount: paddleMount, timeOffset: paddleClockOffsetMS / 1000,
                                                       tagTurns: paddleCalTurns, residualDeg: paddleCalResidualDeg) : nil
        }
        set {
            paddleMount = newValue?.mount ?? []
            paddleClockOffsetMS = (newValue?.timeOffset ?? 0) * 1000
            paddleCalTurns = newValue?.tagTurns ?? 0
            paddleCalResidualDeg = newValue?.residualDeg ?? 0
        }
    }

    /// The exposure cap in seconds, nil when off.
    var exposureCap: Double? { maxExposureMS < 16.5 ? Double(maxExposureMS) / 1000 : nil }

    private static let storageKey = "viewerProfile.v2"

    /// Saved values replaced by the current defaults when a profile older than the version is loaded.
    private static let resetOnUpgrade: [Int: [String]] = [
        3: ["passthrough", "showGrid"],     // XRSLAM 0.5× by default, grid off
        4: ["passthrough"],                 // back to XRSLAM after swipe-switching was removed
        5: Headset.keys,                    // the VRG Pro replaced the Cardboard
        6: ["paddlePrediction"],            // 3D hands at the camera image's moment
        // The first paddle calibrations were learned from tag poses that sometimes flipped behind the
        // camera: dropped, so the paddle calibrates itself again.
        7: ["paddleMount", "paddleClockOffsetMS", "paddleCalTurns", "paddleCalResidualDeg"],
        8: ["paddleHaptics"],               // full-strength hit buzz
        9: ["maxExposureMS"],               // 3 ms: the paddle's tags blur less
    ]

    /// Switch headsets: its optics and eye position replace the current ones.
    mutating func apply(_ preset: Headset) {
        headset = preset
        let o = preset.optics
        interLensMM = o.interLensMM
        screenToLensMM = o.screenToLensMM
        trayToLensMM = o.trayToLensMM
        k1 = o.k1
        k2 = o.k2
        lensFovDeg = o.lensFovDeg
        eyeOffsetZMM = o.eyeOffsetZMM
    }

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

/// Phone headsets with their optics. Cardboard v2 is Google's published profile; the VRG Pro (a generic
/// 42 mm-lens headset with IPD and focus knobs, no published profile) is estimated from its build and
/// is meant to be fine-tuned with the sliders.
enum Headset: String, Codable, CaseIterable {
    case vrgPro
    case cardboardV2

    var label: String {
        switch self {
        case .vrgPro: "VRG Pro"
        case .cardboardV2: "Google Cardboard v2"
        }
    }

    struct Optics {
        var interLensMM: Float
        var screenToLensMM: Float
        var trayToLensMM: Float
        var k1: Float
        var k2: Float
        var lensFovDeg: Float
        var eyeOffsetZMM: Float          // the camera's distance in front of the eyes
    }

    var optics: Optics {
        switch self {
        case .vrgPro:
            // Lens spacing: set the IPD knob to your IPD and this slider to the same. The phone is
            // clamped centred on the lenses (iPhone 15: 71.6 mm tall -> lens axis ~36 mm from its edge).
            Optics(interLensMM: 63.0, screenToLensMM: 42.0, trayToLensMM: 36.0, k1: 0.38, k2: 0.40,
                   lensFovDeg: 55, eyeOffsetZMM: 72)
        case .cardboardV2:
            Optics(interLensMM: 63.9, screenToLensMM: 39.0, trayToLensMM: 35.0, k1: 0.34, k2: 0.55,
                   lensFovDeg: 50, eyeOffsetZMM: 62)
        }
    }

    /// The profile fields a preset sets.
    static let keys = ["headset", "interLensMM", "screenToLensMM", "trayToLensMM", "k1", "k2", "lensFovDeg",
                       "eyeOffsetZMM"]
}

enum GameScene: String, Codable, CaseIterable {
    case pingPong
    case cube           // the tracking test cube
    case none

    var label: String {
        switch self {
        case .pingPong: "Ping pong"
        case .cube: "Test cube"
        case .none: "Nothing"
        }
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
