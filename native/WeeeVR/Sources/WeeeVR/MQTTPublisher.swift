import Foundation
import Network

/// Posts values to an MQTT broker straight from the phone (the game's score record), over plain TCP
/// on port 1883 with the packets in `MQTTPacket`. QoS 0 and retained: the broker keeps the latest
/// value, so anyone subscribing later still sees it.
///
/// Runs on its own queue. It keeps the latest value of each topic and sends them all again after every
/// (re)connect, so a dropped connection loses nothing. It reconnects every 5 s while it can't reach the
/// broker, and pings every 20 s (keep-alive 30 s).
final class MQTTPublisher {
    private let queue = DispatchQueue(label: "mqtt.publisher")
    private var host: String?
    private var port: UInt16 = 1883
    private var connection: NWConnection?
    private var connected = false
    private var generation = 0
    private var pingTimer: DispatchSourceTimer?
    private var topic = ""
    private var latest: String?                     // the latest value, sent again after each (re)connect
    private let clientID = "weeevr-" + String(UInt32.random(in: 0...UInt32.max), radix: 16)

    private let lock = NSLock()
    private var _status = "off"
    private var _lastPosted: String?

    /// A line for the settings: connection state and the last value sent.
    var status: String {
        lock.lock()
        defer { lock.unlock() }
        return _status + (_lastPosted.map { " · last posted \($0)" } ?? "")
    }

    var onMessage: ((String) -> Void)?

    /// Connect to `host` (nil: stop) and post to `topic`. A new topic gets the latest value at once.
    func start(host: String?, topic: String, port: UInt16 = 1883) {
        queue.async { [self] in
            if topic != self.topic {
                self.topic = topic
                if connected, let latest { publishNow(latest) }
            }
            guard host != self.host || port != self.port || (host != nil && connection == nil) else { return }
            tearDown()
            self.host = host?.trimmingCharacters(in: .whitespaces).isEmpty == false ? host : nil
            self.port = port
            if self.host == nil { setStatus("off") } else { open() }
        }
    }

    /// Post `payload` to the topic now (or as soon as connected). Any thread; never blocks.
    func publish(_ payload: String) {
        queue.async { [self] in
            latest = payload
            if connected { publishNow(payload) }
        }
    }

    private func publishNow(_ payload: String) {
        guard !topic.isEmpty else { return }
        send(MQTTPacket.publish(topic: topic, payload: payload, retain: true), posted: "\(payload) to \(topic)")
    }

    // MARK: Queue

    private func open() {
        guard let host else { return }
        generation += 1
        let gen = generation
        let params = NWParameters.tcp
        if let tcp = params.defaultProtocolStack.transportProtocol as? NWProtocolTCP.Options { tcp.connectionTimeout = 5 }
        let conn = NWConnection(host: NWEndpoint.Host(host), port: NWEndpoint.Port(rawValue: port)!, using: params)
        connection = conn
        setStatus("connecting to \(host)…")
        conn.stateUpdateHandler = { [weak self] state in
            guard let self else { return }
            self.queue.async {
                guard gen == self.generation else { return }
                switch state {
                case .ready:
                    conn.send(content: Data(MQTTPacket.connect(clientID: self.clientID, keepAlive: 30)),
                              completion: .contentProcessed { _ in })
                    self.awaitConnectAck(conn, gen: gen)
                case .waiting(let error), .failed(let error):
                    self.setStatus("can't reach \(host): \(error.localizedDescription)")
                    self.retry()
                default:
                    break
                }
            }
        }
        conn.start(queue: queue)
    }

    private func awaitConnectAck(_ conn: NWConnection, gen: Int) {
        conn.receive(minimumIncompleteLength: 4, maximumLength: 4) { [weak self] data, _, _, error in
            guard let self, gen == self.generation else { return }
            guard let data, let code = MQTTPacket.connectAck(Array(data)), code == 0 else {
                self.setStatus("broker refused the connection" + (error.map { ": \($0.localizedDescription)" } ?? ""))
                self.retry()
                return
            }
            self.connected = true
            self.setStatus("connected to \(self.host ?? "?")")
            self.onMessage?("mqtt: connected to \(self.host ?? "?")")
            if let latest = self.latest { self.publishNow(latest) }
            self.startPings()
            self.drain(conn, gen: gen)
        }
    }

    /// Read and ignore what the broker sends (PINGRESP); a closed stream means reconnect.
    private func drain(_ conn: NWConnection, gen: Int) {
        conn.receive(minimumIncompleteLength: 1, maximumLength: 256) { [weak self] data, _, complete, error in
            guard let self, gen == self.generation else { return }
            if complete || error != nil || data == nil {
                self.setStatus("disconnected")
                self.retry()
                return
            }
            self.drain(conn, gen: gen)
        }
    }

    private func startPings() {
        let timer = DispatchSource.makeTimerSource(queue: queue)
        timer.schedule(deadline: .now() + 20, repeating: 20)
        timer.setEventHandler { [weak self] in self?.send(MQTTPacket.pingRequest, posted: nil) }
        timer.resume()
        pingTimer = timer
    }

    private func send(_ bytes: [UInt8], posted: String?) {
        guard let connection else { return }
        connection.send(content: Data(bytes), completion: .contentProcessed { [weak self] error in
            guard let self else { return }
            if let error {
                self.setStatus("send failed: \(error.localizedDescription)")
            } else if let posted {
                self.lock.lock(); self._lastPosted = posted; self.lock.unlock()
                self.onMessage?("mqtt: posted \(posted)")
            }
        })
    }

    private func retry() {
        tearDown()
        let next = generation
        queue.asyncAfter(deadline: .now() + 5) { [self] in
            if next == generation, host != nil, connection == nil { open() }
        }
    }

    private func tearDown() {
        generation += 1
        pingTimer?.cancel()
        pingTimer = nil
        if connected { connection?.send(content: Data(MQTTPacket.disconnect), completion: .idempotent) }
        connection?.cancel()
        connection = nil
        connected = false
    }

    private func setStatus(_ s: String) {
        lock.lock()
        _status = s
        lock.unlock()
    }
}
