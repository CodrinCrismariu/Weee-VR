import Foundation
import QuartzCore
import SwiftUI

@MainActor
final class AppModel: ObservableObject {
    @Published var showHUD = false {     // two-finger tap shows the stats HUD
        didSet { if !showHUD { engine?.setHUD(nil) } }
    }
    @Published var showCalibration = false
    @Published var calibrationResult: String?
    @Published var viewer = ViewerProfile.load() {
        didSet {
            viewer.save()
            engine?.update(viewer: viewer)
        }
    }

    let engine: VREngine?
    private var startupError: String?
    private var timer: Timer?
    private var lastReceived: UInt64 = 0
    private var lastTick = CACurrentMediaTime()
    private var lastLog = 0.0

    init() {
        do {
            engine = try VREngine(viewer: ViewerProfile.load())
        } catch {
            engine = nil
            startupError = error.localizedDescription
            Log.info(String(format: "startup failed: %@", error.localizedDescription))
        }
    }

    func start() {
        guard let engine, timer == nil else { return }
        engine.start()
        timer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated { self?.refresh() }
        }
    }

    // MARK: Gestures

    func resetTracking() { engine?.resetTracking() }
    func toggleHUD() { showHUD.toggle() }
    func toggleReprojection() { viewer.reprojection.toggle() }
    func openCalibration() { showCalibration = true }

    /// Camera-IMU calibration dataset (scripts/calib): ultra-wide gyro mode while recording, then
    /// back to the previous mode.
    func recordCalibration() {
        guard let engine, !engine.isRecordingCalibration else { return }
        let previous = viewer.passthrough
        showCalibration = false
        viewer.passthrough = .ultraWide
        engine.startCalibrationRecording { [weak self] directory in
            self?.viewer.passthrough = previous
            self?.calibrationResult = directory.map { "saved \($0.lastPathComponent)" } ?? "recording FAILED"
        }
    }

    // MARK: HUD + log

    private func refresh() {
        guard let engine else { return }
        let now = CACurrentMediaTime()
        let s = engine.stats.snapshot()
        let status = engine.status()
        let counters = engine.mailbox.counters()
        let cameraFps = Double(counters.received &- lastReceived) / max(now - lastTick, 1e-3)
        lastReceived = counters.received
        lastTick = now
        let thermal = Self.describe(ProcessInfo.processInfo.thermalState)
        let p = status.position
        let memory = Self.memoryFootprintMB()

        if showHUD {
            var lines = [
                "\(viewer.passthrough.label) · \(status.source)",
                String(format: "camera fov %.0f° × %.0f°", status.cameraFovDeg.x, status.cameraFovDeg.y),
                String(format: "%.0f fps · cam %.0f · missed %ld · %@ · %.0f MB", s.displayFps, cameraFps,
                       s.missedRefreshes, thermal, memory),
                String(format: "photon p50 %.1f p95 %.1f max %.1f ms", s.photonP50, s.photonP95, s.photonMax),
                String(format: "cam→app %.1f · cpu %.2f gpu %.2f ms", s.arrivalP50, s.renderCpuP95, s.gpuP95),
                String(format: "depth %.2f m · reproj %@ · latch %@", status.depth,
                       viewer.reprojection ? "on" : "off", viewer.lateLatch ? "on" : "off"),
            ]
            if let tracking = status.tracking {
                lines.append(String(format: "%@ · pos %.2f %.2f %.2f", tracking, p.x, p.y, p.z))
            }
            if let slamInfo = status.slamInfo { lines.append(slamInfo) }
            if let handsInfo = status.handsInfo { lines.append(handsInfo) }
            if let recording = engine.calibrationStatus {
                lines = [recording] + lines
            } else if let calibrationResult {
                lines.insert("calibration \(calibrationResult)", at: 0)
            }
            if let startupError { lines = ["startup failed:", startupError] }
            engine.setHUD(HUDRasterizer.makeTexture(text: lines.joined(separator: "\n"), device: engine.device))
        }

        if now - lastLog >= 2 {
            lastLog = now
            Log.info(String(format: "stats mode=%@ fov=%.0fx%.0f source=%@ tracking=%@ pos=%.2f,%.2f,%.2f depth=%.2f reproj=%ld latch=%ld hud=%ld slam=[%@] hands=[%@] p50=%.1f p95=%.1f max=%.1f arrival=%.1f fps=%.1f cam=%.1f missed=%ld cpu95=%.2f gpu95=%.2f budget=%.1f unshown=%llu thermal=%@ mem=%.0fMB n=%ld",
                  viewer.passthrough.label, status.cameraFovDeg.x, status.cameraFovDeg.y, status.source, status.tracking ?? "-", p.x, p.y, p.z, status.depth,
                  viewer.reprojection ? 1 : 0, viewer.lateLatch ? 1 : 0, showHUD ? 1 : 0, status.slamInfo ?? "-", status.handsInfo ?? "-",
                  s.photonP50, s.photonP95, s.photonMax, s.arrivalP50, s.displayFps, cameraFps,
                  s.missedRefreshes, s.renderCpuP95, s.gpuP95, status.renderBudgetMS,
                  counters.neverShown, thermal, memory, s.samples))
        }
    }

    /// The app's memory footprint (what iOS counts against its memory limit), in MB.
    private static func memoryFootprintMB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return result == KERN_SUCCESS ? Double(info.phys_footprint) / 1_048_576 : 0
    }

    private static func describe(_ state: ProcessInfo.ThermalState) -> String {
        switch state {
        case .nominal: "cool"
        case .fair: "warm"
        case .serious: "HOT"
        case .critical: "CRITICAL"
        @unknown default: "?"
        }
    }
}
