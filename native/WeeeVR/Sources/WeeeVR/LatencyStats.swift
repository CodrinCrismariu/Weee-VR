import Foundation

/// Rolling latency numbers, all in milliseconds:
/// - photon: camera capture timestamp -> the moment the frame was actually on screen (presentedTime)
/// - arrival: camera capture timestamp -> the app receiving the buffer (sensor + ISP + driver)
/// - render CPU / GPU: our own per-frame cost (guards against tracking slowing rendering down)
/// - missed refreshes: gaps of more than 1.5 refresh intervals between presented frames
final class LatencyStats {
    struct Snapshot {
        var photonP50 = 0.0, photonP95 = 0.0, photonMax = 0.0
        var arrivalP50 = 0.0
        var renderCpuP95 = 0.0, gpuP95 = 0.0
        var displayFps = 0.0
        var missedRefreshes = 0
        var samples = 0
    }

    private static let refresh = 1.0 / 60.0

    private let lock = NSLock()
    private var photon: [Double] = []
    private var arrival: [Double] = []
    private var renderCpu: [Double] = []
    private var gpu: [Double] = []
    private var presentTimes: [Double] = []
    private var missed = 0
    private var missedUnread = 0
    private let window = 240

    func addPresent(captureTime: Double, arrivalTime: Double, presentedTime: Double) {
        let photonMS = (presentedTime - captureTime) * 1000
        let arrivalMS = (arrivalTime - captureTime) * 1000
        lock.lock()
        defer { lock.unlock() }
        if let last = presentTimes.last, presentedTime - last > 1.5 * Self.refresh {
            let skipped = Int(((presentedTime - last) / Self.refresh).rounded()) - 1
            missed += skipped
            missedUnread += skipped
        }
        append(&presentTimes, presentedTime)
        guard photonMS > 0, photonMS < 1000 else { return }   // ignore clock glitches
        append(&photon, photonMS)
        append(&arrival, arrivalMS)
    }

    func addRenderCost(cpuSeconds: Double) {
        lock.lock()
        append(&renderCpu, cpuSeconds * 1000)
        lock.unlock()
    }

    func addGpuTime(seconds: Double) {
        lock.lock()
        append(&gpu, seconds * 1000)
        lock.unlock()
    }

    /// Most recent GPU frame time in seconds (for the render loop's time budget).
    var snapshotGpuEstimate: Double {
        lock.lock()
        defer { lock.unlock() }
        return (gpu.last ?? 0) / 1000
    }

    /// Refreshes missed since the previous call (the render loop widens its time budget on a miss).
    func takeNewMisses() -> Int {
        lock.lock()
        defer { lock.unlock() }
        let n = missedUnread
        missedUnread = 0
        return n
    }

    func reset() {
        lock.lock()
        photon.removeAll(); arrival.removeAll(); renderCpu.removeAll(); gpu.removeAll(); presentTimes.removeAll()
        missed = 0
        missedUnread = 0
        lock.unlock()
    }

    func snapshot() -> Snapshot {
        lock.lock()
        let photon = self.photon, arrival = self.arrival, presents = self.presentTimes
        let renderCpu = self.renderCpu, gpu = self.gpu, missed = self.missed
        lock.unlock()

        var s = Snapshot()
        s.samples = photon.count
        s.missedRefreshes = missed
        if !photon.isEmpty {
            let sorted = photon.sorted()
            s.photonP50 = Self.percentile(sorted, 0.5)
            s.photonP95 = Self.percentile(sorted, 0.95)
            s.photonMax = sorted.last ?? 0
            s.arrivalP50 = Self.percentile(arrival.sorted(), 0.5)
        }
        s.renderCpuP95 = Self.percentile(renderCpu.sorted(), 0.95)
        s.gpuP95 = Self.percentile(gpu.sorted(), 0.95)
        if presents.count > 1, let first = presents.first, let last = presents.last, last > first {
            s.displayFps = Double(presents.count - 1) / (last - first)
        }
        return s
    }

    private func append(_ array: inout [Double], _ value: Double) {
        array.append(value)
        if array.count > window { array.removeFirst() }
    }

    private static func percentile(_ sorted: [Double], _ p: Double) -> Double {
        guard !sorted.isEmpty else { return 0 }
        return sorted[min(sorted.count - 1, Int(Double(sorted.count - 1) * p + 0.5))]
    }
}
