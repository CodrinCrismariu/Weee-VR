import SwiftUI
import UIKit

struct RootView: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        // The HUD is drawn inside the Metal pass (HUDRasterizer), not as a SwiftUI overlay.
        VRContainer(model: model)
            .ignoresSafeArea()
        .statusBarHidden(true)
        .persistentSystemOverlays(.hidden)
        .sheet(isPresented: $model.showCalibration) {
            CalibrationView()
                .environmentObject(model)
        }
        .onAppear {
            UIApplication.shared.isIdleTimerDisabled = true
            model.start()
        }
    }
}

struct CalibrationView: View {
    @EnvironmentObject private var model: AppModel
    @Environment(\.dismiss) private var dismiss
    @State private var paddleHost = ""
    @State private var mqttHost = ""
    @State private var mqttTopic = ""

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    SliderRow(title: "Vibration strength", value: $model.viewer.paddleHaptics, range: 0...1,
                              format: "%.0f%%", scale: 100)
                    Button("Test buzz (a medium hit)") { model.testPaddleBuzz() }
                } header: {
                    Text("Paddle vibration")
                } footer: {
                    Text("Every ball hit buzzes the paddle: a soft touch short and light, a hard hit long and full power. 0% turns it off.")
                }
                Section("Tracking") {
                    Picker("Camera + tracking", selection: $model.viewer.passthrough) {
                        ForEach(Passthrough.allCases, id: \.self) { Text($0.label).tag($0) }
                    }
                    Toggle("Late-latch render", isOn: $model.viewer.lateLatch)
                    Toggle("Auto depth (ARKit feature points)", isOn: $model.viewer.autoDepth)
                    SliderRow(title: "Fixed depth", value: $model.viewer.fixedDepthM, range: 0.3...8, format: "%.2f m")
                }
                Section {
                    Button("Record camera calibration (75 s)") { model.recordCalibration() }
                } header: {
                    Text("Camera + IMU calibration")
                } footer: {
                    Text("Point the back camera at the grid on the laptop screen. 3 s countdown, then 75 s: first slowly move the grid through every part of the view, then tilt, turn and shift the phone in all directions while keeping the grid in view.")
                }
                Section {
                    Picker("Scene (placed on tap)", selection: $model.viewer.scene) {
                        ForEach(GameScene.allCases, id: \.self) { Text($0.label).tag($0) }
                    }
                    SliderRow(title: "Table below eyes", value: $model.viewer.tableDropM, range: 0.35...1.0, format: "%.2f m")
                    SliderRow(title: "Assist", value: $model.viewer.assist, range: 0...1, format: "%.0f%%", scale: 100)
                    if model.gameRecording {
                        Button("Stop recording the game", role: .destructive) { model.stopGameRecording() }
                    } else {
                        Button("Record a game to 11 (video)") { model.recordGame() }
                    }
                    if let result = model.gameRecordingResult {
                        Text(result).font(.footnote).foregroundStyle(.secondary)
                    }
                    Toggle("Table on its tag (ID 67)", isOn: $model.viewer.tableTag)
                    Picker("Tag 67's label faces", selection: $model.viewer.tableTagLabel) {
                        Text("Me").tag(0)
                        Text("The far end").tag(1)
                        Text("My left").tag(2)
                        Text("My right").tag(3)
                    }
                    Toggle("Floating serve ball", isOn: $model.viewer.floatingServe)
                    SliderRow(title: "Opponent level", value: $model.viewer.opponentLevel, range: 0...1, format: "%.2f")
                    SliderRow(title: "Sound volume", value: $model.viewer.gameVolume, range: 0...1, format: "%.2f")
                    SliderRow(title: "Hand lead (ahead of camera image)", value: $model.viewer.paddlePrediction, range: 0...0.12, format: "%.3f s")
                    SliderRow(title: "Paddle grip angle", value: $model.viewer.gripAngleDeg, range: 0...80, format: "%.0f°")
                    SliderRow(title: "Paddle grip offset", value: $model.viewer.gripOffsetMM, range: -40...20, format: "%.0f mm")
                    SliderRow(title: "Camera max exposure (less blur)", value: $model.viewer.maxExposureMS, range: 2...17,
                              format: model.viewer.maxExposureMS < 16.5 ? "%.0f ms" : "off")
                } header: {
                    Text("Game")
                } footer: {
                    Text("Tap places the table ahead of you. Pinch the ball (thumb + index) with your left hand to pick it up, open to toss; the paddle is in your right hand. Assist: a bigger hit zone, shots steered onto the table, a slower ball, a clean toss and friendlier returns (0% = realistic). Table height: standing ~0.8 m below the eyes, seated ~0.45 m. Score: first to 11, by 2; you serve every point. Record a game: the score restarts at 0-0 and the video (what the passthrough shows, seen from between your eyes, with the game's sounds) stops a few seconds after the game ends; it goes to Photos and Files > Weee VR > games.")
                }
                Section {
                    Toggle("Post score over MQTT", isOn: $model.viewer.mqttEnabled)
                    LabeledContent("Broker") {
                        TextField("test.mosquitto.org", text: $mqttHost)
                            .multilineTextAlignment(.trailing)
                            .textInputAutocapitalization(.never)
                            .autocorrectionDisabled()
                            .onSubmit { model.viewer.mqttHost = mqttHost }
                    }
                    LabeledContent("Topic") {
                        TextField("ME193/Rogers/Codrin", text: $mqttTopic)
                            .multilineTextAlignment(.trailing)
                            .textInputAutocapitalization(.never)
                            .autocorrectionDisabled()
                            .onSubmit { model.viewer.mqttTopic = mqttTopic }
                    }
                } header: {
                    Text("Score posting (MQTT)")
                } footer: {
                    Text("\(model.mqttStatus)\n\nPosts the record number of continuous hits (my hits in a row that land on the AI's half) as a float, retained, on port 1883.")
                }
                Section {
                    Toggle("Smart paddle", isOn: $model.viewer.smartPaddle)
                    Toggle("Paddle on the camera image", isOn: $model.viewer.paddleOnCameraImage)
                    SliderRow(title: "Blade width", value: $model.viewer.paddleBladeWidthMM, range: 120...220, format: "%.0f mm")
                    SliderRow(title: "Blade length", value: $model.viewer.paddleBladeLengthMM, range: 120...240, format: "%.0f mm")
                    SliderRow(title: "Handle length", value: $model.viewer.paddleHandleMM, range: 60...160, format: "%.0f mm")
                    LabeledContent("Board address") {
                        TextField("10.243.94.25", text: $paddleHost)
                            .multilineTextAlignment(.trailing)
                            .keyboardType(.numbersAndPunctuation)
                            .textInputAutocapitalization(.never)
                            .autocorrectionDisabled()
                            .onSubmit { model.viewer.paddleHost = paddleHost }
                    }
                    if let ends = model.paddleRecordingEnds, ends > Date() {
                        Button("Stop recording and save (\(Int(ends.timeIntervalSinceNow)) s left)") {
                            model.stopPaddleRecording()
                        }
                    } else {
                        Menu("Record paddle data (for tuning)") {
                            Button("20 seconds") { model.recordPaddle(seconds: 20) }
                            Button("1 minute") { model.recordPaddle(seconds: 60) }
                            Button("3 minutes") { model.recordPaddle(seconds: 180) }
                            Button("5 minutes") { model.recordPaddle(seconds: 300) }
                        }
                    }
                    Button("Calibrate IMU (paddle lying still, once)") { model.calibratePaddleIMU() }
                    Button(model.viewer.paddleCalibration == nil ? "Calibrate paddle (once)" : "Recalibrate paddle") {
                        model.calibratePaddle()
                    }
                    if let cal = model.viewer.paddleCalibration {
                        LabeledContent("Calibration", value: String(format: "fit %.1f° · clock %+.0f ms", cal.residualDeg,
                                                                    cal.timeOffset * 1000))
                        Button("Forget paddle calibration", role: .destructive) { model.forgetPaddleCalibration() }
                    }
                    Picker("Tag turned on the blade", selection: $model.viewer.paddleTagTurns) {
                        Text("As calibrated").tag(-1)
                        Text("0°").tag(0)
                        Text("90°").tag(1)
                        Text("180°").tag(2)
                        Text("270°").tag(3)
                    }
                } header: {
                    Text("Smart paddle")
                } footer: {
                    Text("\(model.paddleStatus)\n\nBoth are needed once and then saved. Calibrate IMU: lay the paddle down still for 2 s. Calibrate paddle: hold it in your right hand, tag toward the camera, and slowly turn it about different axes for ~5 s (it buzzes when it starts and twice when done). Without a connected, calibrated paddle the game uses your hand.")
                }
                Section {
                    Toggle("Accurate colour (BT.709 + P3)", isOn: $model.viewer.accurateColor)
                    SliderRow(title: "Sharpening", value: $model.viewer.sharpening, range: 0...1, format: "%.2f")
                    SliderRow(title: "Lens colour fringe fix", value: $model.viewer.chromaticAberration, range: -0.03...0.03, format: "%+.3f")
                    SliderRow(title: "Black lift", value: $model.viewer.blackLift, range: 0...0.06, format: "%.3f")
                    SliderRow(title: "Lens edge softness", value: $model.viewer.edgeSoftness, range: 0...0.3, format: "%.2f")
                    Toggle("Full brightness in VR", isOn: $model.viewer.maxBrightness)
                } header: {
                    Text("Image")
                } footer: {
                    Text("Fringe fix: look at a sharp dark/bright edge near the rim of a lens and slide until its red and blue fringes disappear.")
                }
                Section("Hands") {
                    Toggle("Hand tracking", isOn: $model.viewer.handTracking)
                    Toggle("Show 3D hands", isOn: $model.viewer.handOverlay)
                    SliderRow(title: "Hand size", value: $model.viewer.handScale, range: 0.8...1.2, format: "%.2f×")
                }
                Section("XRSLAM trims on top of the calibration · changes restart tracking") {
                    SliderRow(title: "Camera time offset", value: $model.viewer.slamTimeOffsetMS, range: -30...30, format: "%.1f ms")
                    SliderRow(title: "Camera from IMU x", value: $model.viewer.slamLeverXMM, range: -60...60, format: "%.0f mm")
                    SliderRow(title: "Camera from IMU y", value: $model.viewer.slamLeverYMM, range: -60...60, format: "%.0f mm")
                    SliderRow(title: "Camera from IMU z", value: $model.viewer.slamLeverZMM, range: -60...60, format: "%.0f mm")
                    SliderRow(title: "Frames to tracker", value: $model.viewer.slamFeedHz, range: 15...60, format: "%.0f /s")
                    SliderRow(title: "Keypoints", value: $model.viewer.slamKeypoints, range: 100...400, format: "%.0f")
                    SliderRow(title: "Window keyframes", value: $model.viewer.slamWindow, range: 4...10, format: "%.0f")
                }
                Section("Eyes relative to camera") {
                    SliderRow(title: "Right (+) / left", value: $model.viewer.eyeOffsetXMM, range: -20...100, format: "%.0f mm")
                    SliderRow(title: "Up (+) / down", value: $model.viewer.eyeOffsetYMM, range: -50...50, format: "%.0f mm")
                    SliderRow(title: "Behind camera", value: $model.viewer.eyeOffsetZMM, range: 0...120, format: "%.0f mm")
                }
                Section("Passthrough") {
                    Toggle("Reprojection to display time", isOn: $model.viewer.reprojection)
                    SliderRow(title: "Prediction past display time", value: $model.viewer.predictionMS, range: 0...20, format: "%.1f ms")
                    SliderRow(title: "FOV scale (1 = life-size)", value: $model.viewer.fovScale, range: 0.5...2.0, format: "%.2f")
                    Toggle("Flip camera 180°", isOn: $model.viewer.flipCamera)
                    Toggle("World grid", isOn: $model.viewer.showGrid)
                }
                Section {
                    Picker("Headset", selection: Binding(get: { model.viewer.headset },
                                                         set: { model.viewer.apply($0) })) {
                        ForEach(Headset.allCases, id: \.self) { Text($0.label).tag($0) }
                    }
                    SliderRow(title: "Distortion k1", value: $model.viewer.k1, range: -0.2...1.0, format: "%.3f")
                    SliderRow(title: "Distortion k2", value: $model.viewer.k2, range: -0.2...1.0, format: "%.3f")
                    SliderRow(title: "Lens separation", value: $model.viewer.interLensMM, range: 50...75, format: "%.1f mm")
                    SliderRow(title: "Screen to lens", value: $model.viewer.screenToLensMM, range: 25...55, format: "%.1f mm")
                    SliderRow(title: "Tray to lens centre", value: $model.viewer.trayToLensMM, range: 20...50, format: "%.1f mm")
                    SliderRow(title: "Lens field of view", value: $model.viewer.lensFovDeg, range: 30...70, format: "%.0f°")
                } header: {
                    Text("Headset lenses")
                } footer: {
                    Text("Choosing a headset loads its lens profile; the sliders fine-tune it. VRG Pro: set its IPD knob to your eye distance and Lens separation to the same value. If straight edges bend when you turn your head, adjust k1: edges bulging outward, lower it; pinched inward, raise it. Recenter in the headset: hold your open left palm toward your face for a second.")
                }
                Section("Phone + you") {
                    SliderRow(title: "Bottom bezel", value: $model.viewer.bottomBezelMM, range: 0...10, format: "%.1f mm")
                    SliderRow(title: "Screen density", value: $model.viewer.screenPPI, range: 400...520, format: "%.0f ppi")
                    SliderRow(title: "Your IPD", value: $model.viewer.ipdMM, range: 54...74, format: "%.1f mm")
                }
                Section {
                    Button("Reset everything to defaults (VRG Pro)", role: .destructive) { model.viewer = ViewerProfile() }
                }
            }
            .onAppear {
                paddleHost = model.viewer.paddleHost
                mqttHost = model.viewer.mqttHost
                mqttTopic = model.viewer.mqttTopic
            }
            .navigationTitle("Calibration")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } }
            }
        }
    }
}

private struct SliderRow: View {
    let title: String
    @Binding var value: Float
    let range: ClosedRange<Float>
    let format: String
    var scale: Float = 1                // shown value = value * scale (e.g. 100 for percent)

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(title)
                Spacer()
                Text(String(format: format, value * scale)).monospacedDigit().foregroundStyle(.secondary)
            }
            Slider(value: $value, in: range)
        }
    }
}
