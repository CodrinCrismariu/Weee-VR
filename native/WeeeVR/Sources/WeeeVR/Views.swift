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

    var body: some View {
        NavigationStack {
            Form {
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
                    Toggle("Test cube", isOn: $model.viewer.showCube)
                }
                Section("Cardboard lenses") {
                    SliderRow(title: "Distortion k1", value: $model.viewer.k1, range: -0.2...1.0, format: "%.3f")
                    SliderRow(title: "Distortion k2", value: $model.viewer.k2, range: -0.2...1.0, format: "%.3f")
                    SliderRow(title: "Lens separation", value: $model.viewer.interLensMM, range: 50...75, format: "%.1f mm")
                    SliderRow(title: "Screen to lens", value: $model.viewer.screenToLensMM, range: 25...55, format: "%.1f mm")
                    SliderRow(title: "Tray to lens centre", value: $model.viewer.trayToLensMM, range: 20...50, format: "%.1f mm")
                    SliderRow(title: "Lens field of view", value: $model.viewer.lensFovDeg, range: 30...70, format: "%.0f°")
                }
                Section("Phone + you") {
                    SliderRow(title: "Bottom bezel", value: $model.viewer.bottomBezelMM, range: 0...10, format: "%.1f mm")
                    SliderRow(title: "Screen density", value: $model.viewer.screenPPI, range: 400...520, format: "%.0f ppi")
                    SliderRow(title: "Your IPD", value: $model.viewer.ipdMM, range: 54...74, format: "%.1f mm")
                }
                Section {
                    Button("Reset to Cardboard v2 defaults", role: .destructive) { model.viewer = ViewerProfile() }
                }
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

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(title)
                Spacer()
                Text(String(format: format, value)).monospacedDigit().foregroundStyle(.secondary)
            }
            Slider(value: $value, in: range)
        }
    }
}
