import Darwin
import Foundation
import Network
import QuartzCore

/// The smart paddle (ArduinoApps/smart-paddle on an Arduino UNO Q, on the same Wi-Fi): its IMU
/// stream, and haptic buzzes and calibration requests back.
///
/// The phone connects to the board's TCP port 7700 and says hello with the UDP port it listens on
/// and its own address; the board then sends each 200 Hz sample as one UDP datagram (28 bytes).
/// A ping every 0.25 s keeps the board streaming (it stops after 3 s of silence) and measures the
/// round trip.
///
/// Sample times: the MCU's microsecond clock, mapped to the host clock by the smallest
/// (arrival - MCU time) seen over the last few seconds. The fastest packets bound the link's
/// latency, so the result is each sample's time plus that minimum latency (~5 ms, a constant the
/// paddle calibration measures as its clock offset).
final class PaddleLink {
    static let port: UInt16 = 7700

    struct Sample {
        var time: Double            // host clock (CACurrentMediaTime), as mapped above
        var orientation: SIMD4<Double>   // w x y z: IMU earth <- IMU (earth z up, heading arbitrary)
        var rate: SIMD3<Double>     // rad/s, IMU frame
        var still: Bool
        var calibrated: Bool        // the board has a gyro bias
        var accelNorm: Double       // g
    }

    struct Stats {
        var connected = false
        var streaming = false
        var hz = 0.0
        var ageMS = 0.0             // newest sample's age when read
        var lossPercent = 0.0
        var rttMS = 0.0
        var imuCalibrated: Bool?
        var message = "off"         // latest board/status line
    }

    var onMessage: ((String) -> Void)?

    private let queue = DispatchQueue(label: "paddle.link", qos: .userInteractive)
    private let lock = NSLock()
    private var host: String?
    private var connection: NWConnection?
    private var socketFD: Int32 = -1
    private var udpPort: UInt16 = 0
    private var generation = 0
    private var lineBuffer = Data()
    private var pingTimer: DispatchSourceTimer?
    private var pingsSent: [Int: Double] = [:]
    private var pingCounter = 0

    // Under `lock`
    private var samples: [Sample] = []
    private var readIndex = 0                    // samples handed out by `take`
    private var stats = Stats()
    private var offsets: [(arrival: Double, offset: Double)] = []
    private var minOffset: Double?
    private var mcuEpoch: (last: UInt32, wraps: Double)?
    private var lastSeq: UInt32?
    private var received = 0, lost = 0
    private var rateCount = 0
    private var rateStart = CACurrentMediaTime()
    private var receiverToken = 0                // a receiver thread keeps going while its token is current

    // MARK: Lifecycle

    /// Connect to the board at `host` (nil: disconnect).
    func connect(to host: String?) {
        queue.async { [self] in
            let host = host?.trimmingCharacters(in: .whitespaces)
            guard host != self.host || connection == nil else { return }
            tearDown()
            self.host = (host?.isEmpty ?? true) ? nil : host
            if self.host != nil { open() } else { setMessage("off") }
        }
    }

    func buzz(strength: Double, milliseconds: Int, motors: Int = 3) {
        let pwm = Int((min(max(strength, 0), 1) * 255).rounded())
        send("buzz \(motors) \(pwm) \(min(max(milliseconds, 0), 250))")
    }

    /// Measure the gyro bias (the paddle must lie still for ~2 s); the board keeps it for every start.
    func calibrateIMU() { send("calibrate 2000") }

    /// Samples received since the last call (oldest first).
    func take() -> [Sample] {
        lock.lock()
        defer { lock.unlock() }
        let new = Array(samples[min(readIndex, samples.count)...])
        readIndex = samples.count
        return new
    }

    var currentStats: Stats {
        lock.lock()
        defer { lock.unlock() }
        var s = stats
        if let last = samples.last { s.ageMS = (CACurrentMediaTime() - last.time) * 1000 }
        s.streaming = s.connected && s.ageMS < 500
        return s
    }

    // MARK: Connection (queue)

    private func open() {
        guard let host else { return }
        generation += 1
        let gen = generation
        guard openSocket() else {
            setMessage("no UDP socket")
            return
        }
        let params = NWParameters.tcp
        if let tcp = params.defaultProtocolStack.transportProtocol as? NWProtocolTCP.Options {
            tcp.noDelay = true
            tcp.connectionTimeout = 3
        }
        params.serviceClass = .interactiveVoice
        let conn = NWConnection(host: NWEndpoint.Host(host), port: NWEndpoint.Port(rawValue: Self.port)!, using: params)
        connection = conn
        setMessage("connecting to \(host)…")
        conn.stateUpdateHandler = { [weak self] state in
            guard let self else { return }
            self.queue.async {
                guard gen == self.generation else { return }
                switch state {
                case .ready:
                    self.lock.lock(); self.stats.connected = true; self.lock.unlock()
                    var localIP = ""
                    if case .hostPort(let local, _)? = conn.currentPath?.localEndpoint { localIP = "\(local)" }
                    if let percent = localIP.firstIndex(of: "%") { localIP = String(localIP[..<percent]) }
                    self.sendNow("hello \(self.udpPort) \(localIP)")
                    self.setMessage("connected to \(host)")
                    self.startPings()
                    self.receiveLines(conn, gen: gen)
                case .waiting(let error):
                    self.setMessage("can't reach \(host): \(error.localizedDescription)")
                    self.retry(gen)
                case .failed(let error):
                    self.setMessage("link failed: \(error.localizedDescription)")
                    self.retry(gen)
                default:
                    break
                }
            }
        }
        conn.start(queue: queue)
    }

    private func retry(_ gen: Int) {
        tearDown()
        let next = generation
        queue.asyncAfter(deadline: .now() + 2) { [self] in
            if next == generation, host != nil, connection == nil { open() }
        }
    }

    private func tearDown() {
        generation += 1
        pingTimer?.cancel()
        pingTimer = nil
        connection?.cancel()
        connection = nil
        lineBuffer.removeAll()
        lock.lock()
        receiverToken += 1          // the receiver thread stops and closes its socket itself
        if socketFD >= 0 { shutdown(socketFD, SHUT_RDWR) }
        socketFD = -1
        stats.connected = false
        offsets.removeAll()
        minOffset = nil
        mcuEpoch = nil
        lastSeq = nil
        lock.unlock()
    }

    private func send(_ line: String) {
        queue.async { [self] in sendNow(line) }
    }

    private func sendNow(_ line: String) {
        connection?.send(content: (line + "\n").data(using: .utf8), completion: .contentProcessed { _ in })
    }

    private func startPings() {
        let timer = DispatchSource.makeTimerSource(queue: queue)
        timer.schedule(deadline: .now() + 0.25, repeating: 0.25)
        timer.setEventHandler { [weak self] in
            guard let self else { return }
            pingCounter += 1
            pingsSent[pingCounter] = CACurrentMediaTime()
            pingsSent = pingsSent.filter { CACurrentMediaTime() - $0.value < 5 }
            sendNow("ping \(pingCounter)")
        }
        timer.resume()
        pingTimer = timer
    }

    private func receiveLines(_ conn: NWConnection, gen: Int) {
        conn.receive(minimumIncompleteLength: 1, maximumLength: 4096) { [weak self] data, _, complete, error in
            guard let self else { return }
            self.queue.async {
                guard gen == self.generation else { return }
                if let data { self.lineBuffer.append(data) }
                while let newline = self.lineBuffer.firstIndex(of: 0x0A) {
                    let line = String(decoding: self.lineBuffer[..<newline], as: UTF8.self)
                    self.lineBuffer.removeSubrange(...newline)
                    self.handle(line)
                }
                if complete || error != nil {
                    self.setMessage("board closed the link")
                    self.retry(gen)
                } else {
                    self.receiveLines(conn, gen: gen)
                }
            }
        }
    }

    private func handle(_ line: String) {
        let words = line.split(separator: " ").map(String.init)
        guard let first = words.first else { return }
        switch first {
        case "pong":
            if words.count > 1, let n = Int(words[1]), let sent = pingsSent.removeValue(forKey: n) {
                lock.lock()
                stats.rttMS += ((CACurrentMediaTime() - sent) * 1000 - stats.rttMS) * 0.2
                lock.unlock()
            }
        case "bias":
            lock.lock(); stats.imuCalibrated = true; lock.unlock()
            setMessage("IMU calibrated: gyro bias \(words.dropFirst().joined(separator: " ")) °/s")
        case "uncalibrated":
            lock.lock(); stats.imuCalibrated = false; lock.unlock()
            setMessage("IMU not calibrated yet: lay the paddle still and press Calibrate IMU")
        case "calfail":
            setMessage("IMU calibration failed (\(words.dropFirst().joined(separator: " "))): keep the paddle still")
        case "status":
            let text = words.dropFirst().joined(separator: " ")
            if !text.hasPrefix("imu ") { setMessage("board: \(text)") }     // skip the 5 s rate reports
        case "ok":
            break
        default:
            setMessage("board: \(line)")
        }
    }

    private func setMessage(_ text: String) {
        lock.lock()
        stats.message = text
        lock.unlock()
        onMessage?(text)
    }

    // MARK: UDP (receiver thread)

    private func openSocket() -> Bool {
        let fd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)
        guard fd >= 0 else { return false }
        var addr = sockaddr_in()
        addr.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = 0
        addr.sin_addr.s_addr = INADDR_ANY
        let bound = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) }
        }
        guard bound == 0 else { close(fd); return false }
        var len = socklen_t(MemoryLayout<sockaddr_in>.size)
        _ = withUnsafeMutablePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { getsockname(fd, $0, &len) }
        }
        udpPort = UInt16(bigEndian: addr.sin_port)
        var tos: Int32 = 0xB8                       // DSCP EF (Wi-Fi voice queue)
        setsockopt(fd, IPPROTO_IP, IP_TOS, &tos, socklen_t(MemoryLayout<Int32>.size))
        var timeout = timeval(tv_sec: 0, tv_usec: 300_000)
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, socklen_t(MemoryLayout<timeval>.size))
        lock.lock()
        socketFD = fd
        let token = receiverToken
        lock.unlock()
        let thread = Thread { [weak self] in
            self?.receiveLoop(fd, token: token)
            close(fd)
        }
        thread.name = "paddle.udp"
        thread.qualityOfService = .userInteractive
        thread.start()
        return true
    }

    private func receiveLoop(_ fd: Int32, token: Int) {
        var buffer = [UInt8](repeating: 0, count: 64)
        while true {
            let n = buffer.withUnsafeMutableBytes { recv(fd, $0.baseAddress, 64, 0) }
            lock.lock()
            let current = token == receiverToken
            lock.unlock()
            if !current { return }
            if n <= 0 {
                if n < 0, errno == EAGAIN || errno == EINTR { continue }
                return
            }
            let arrival = CACurrentMediaTime()
            guard n == 28, buffer[0] == 0x57, buffer[1] == 0x50, buffer[2] == 0x49, buffer[3] == 0x31 else { continue }
            func word(_ i: Int) -> UInt32 {
                UInt32(buffer[4 + 4 * i]) | UInt32(buffer[5 + 4 * i]) << 8 | UInt32(buffer[6 + 4 * i]) << 16
                    | UInt32(buffer[7 + 4 * i]) << 24
            }
            func s16(_ v: UInt32) -> Double { Double(Int16(bitPattern: UInt16(truncatingIfNeeded: v))) }
            let seq = word(0), t = word(1), q01 = word(2), q23 = word(3), w01 = word(4), w2f = word(5)
            let q = SIMD4(s16(q01), s16(q01 >> 16), s16(q23), s16(q23 >> 16)) / 32767
            let rate = SIMD3(s16(w01), s16(w01 >> 16), s16(w2f)) / 16 * .pi / 180
            let flags = (w2f >> 16) & 0xFF
            ingest(seq: seq, mcu: t, arrival: arrival, sample: Sample(
                time: 0, orientation: q / max((q * q).sum().squareRoot(), 1e-9), rate: rate,
                still: flags & 1 != 0, calibrated: flags & 2 != 0, accelNorm: Double(w2f >> 24) / 16))
        }
    }

    private func ingest(seq: UInt32, mcu: UInt32, arrival: Double, sample: Sample) {
        lock.lock()
        defer { lock.unlock() }
        // MCU seconds, unwrapped across the 32-bit microsecond rollover (~71 min).
        var wraps = mcuEpoch?.wraps ?? 0
        if let last = mcuEpoch?.last, mcu < last, last - mcu > 0x8000_0000 { wraps += 4294.967296 }
        mcuEpoch = (mcu, wraps)
        let mcuSeconds = Double(mcu) / 1e6 + wraps

        // Host <- MCU: the smallest offset over the last 5 s.
        let offset = arrival - mcuSeconds
        offsets.append((arrival, offset))
        if offsets.count > 1200 { offsets.removeFirst(offsets.count - 1000) }
        offsets.removeAll { arrival - $0.arrival > 5 }
        if minOffset == nil || offset < minOffset! {
            minOffset = offset
        } else if received % 50 == 0 {
            minOffset = offsets.map(\.offset).min()
        }
        // A restarted MCU (time went backwards a lot): start over.
        if let m = minOffset, offset - m > 2 {
            offsets = [(arrival, offset)]
            minOffset = offset
            samples.removeAll()
            readIndex = 0
        }

        if let lastSeq, seq &- lastSeq > 1, seq &- lastSeq < 1000 { lost += Int(seq &- lastSeq) - 1 }
        lastSeq = seq
        received += 1
        var s = sample
        s.time = mcuSeconds + (minOffset ?? offset)
        if let last = samples.last, s.time <= last.time { return }
        samples.append(s)
        if samples.count > 800 {
            let drop = samples.count - 600
            samples.removeFirst(drop)
            readIndex = max(readIndex - drop, 0)
        }
        stats.imuCalibrated = s.calibrated

        rateCount += 1
        if arrival - rateStart >= 1 {
            stats.hz = Double(rateCount) / (arrival - rateStart)
            stats.lossPercent = received + lost > 0 ? 100 * Double(lost) / Double(received + lost) : 0
            rateCount = 0
            rateStart = arrival
            received = 0
            lost = 0
        }
    }
}
