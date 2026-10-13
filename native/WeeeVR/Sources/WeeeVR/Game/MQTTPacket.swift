import Foundation

/// The few MQTT 3.1.1 packets the score publisher needs (connect, publish at QoS 0, ping, disconnect),
/// encoded by hand: a fixed header byte, the "remaining length" as a base-128 varint, then the body.
/// Strings are a 2-byte big-endian length and UTF-8. Kept free of Network.framework so the Linux
/// self-test can check the bytes.
enum MQTTPacket {
    static func remainingLength(_ n: Int) -> [UInt8] {
        var n = n, out: [UInt8] = []
        repeat {
            var byte = UInt8(n % 128)
            n /= 128
            if n > 0 { byte |= 0x80 }
            out.append(byte)
        } while n > 0
        return out
    }

    static func string(_ s: String) -> [UInt8] {
        let bytes = Array(s.utf8)
        return [UInt8(bytes.count >> 8), UInt8(bytes.count & 0xFF)] + bytes
    }

    /// CONNECT: protocol "MQTT" level 4, clean session, no will, no login.
    static func connect(clientID: String, keepAlive: UInt16) -> [UInt8] {
        let body = string("MQTT") + [4, 0x02, UInt8(keepAlive >> 8), UInt8(keepAlive & 0xFF)] + string(clientID)
        return [0x10] + remainingLength(body.count) + body
    }

    /// PUBLISH at QoS 0 (no packet id), optionally retained: the broker keeps the last value for late
    /// subscribers.
    static func publish(topic: String, payload: String, retain: Bool) -> [UInt8] {
        let body = string(topic) + Array(payload.utf8)
        return [0x30 | (retain ? 0x01 : 0)] + remainingLength(body.count) + body
    }

    static let pingRequest: [UInt8] = [0xC0, 0x00]
    static let disconnect: [UInt8] = [0xE0, 0x00]

    /// CONNACK's return code (0 = accepted), if `bytes` start with one.
    static func connectAck(_ bytes: [UInt8]) -> UInt8? {
        guard bytes.count >= 4, bytes[0] == 0x20, bytes[1] == 0x02 else { return nil }
        return bytes[3]
    }
}
