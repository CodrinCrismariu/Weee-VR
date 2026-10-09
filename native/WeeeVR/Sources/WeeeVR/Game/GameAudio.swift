import AVFoundation
import Foundation

/// The game's contact sounds, synthesised at launch (no asset files) and played from where the ball
/// is: an `AVAudioEnvironmentNode` with the listener following the head. Everything runs on its own
/// queue; the render thread only enqueues.
final class GameAudio {
    private let queue = DispatchQueue(label: "game.audio", qos: .userInitiated)
    private let engine = AVAudioEngine()
    private let environment = AVAudioEnvironmentNode()
    private var players: [AVAudioPlayerNode] = []
    private var buffers: [AVAudioPCMBuffer] = []
    private var next = 0
    private var running = false
    private var _volume: Float = 0.8

    init() {
        queue.async { [self] in setUp() }
    }

    var volume: Float {
        get { queue.sync { _volume } }
        set { queue.async { [self] in _volume = newValue } }
    }

    /// `kind` at world position `p`, louder for faster impacts (m/s).
    func play(_ kind: GameSound, at p: SIMD3<Float>, speed: Double) {
        queue.async { [self] in
            guard running, !players.isEmpty else { return }
            let player = players[next]
            next = (next + 1) % players.count
            player.position = AVAudio3DPoint(x: p.x, y: p.y, z: p.z)
            player.volume = _volume * Float(min(max(speed / 5, 0.08), 1))
            player.scheduleBuffer(buffers[kind.rawValue], at: nil, options: .interrupts)
            if !player.isPlaying { player.play() }
        }
    }

    /// The listener: head position and orientation (world, y up).
    func setListener(position: SIMD3<Float>, forward: SIMD3<Float>, up: SIMD3<Float>) {
        queue.async { [self] in
            guard running else { return }
            environment.listenerPosition = AVAudio3DPoint(x: position.x, y: position.y, z: position.z)
            environment.listenerVectorOrientation = AVAudio3DVectorOrientation(
                forward: AVAudio3DVector(x: forward.x, y: forward.y, z: forward.z),
                up: AVAudio3DVector(x: up.x, y: up.y, z: up.z))
        }
    }

    private func setUp() {
        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playback, options: [.mixWithOthers])
            try session.setPreferredIOBufferDuration(0.005)
            try session.setActive(true)

            guard let format = AVAudioFormat(standardFormatWithSampleRate: 48_000, channels: 1) else { return }
            engine.attach(environment)
            engine.connect(environment, to: engine.mainMixerNode, format: nil)
            environment.distanceAttenuationParameters.distanceAttenuationModel = .inverse
            environment.distanceAttenuationParameters.referenceDistance = 0.6
            environment.distanceAttenuationParameters.maximumDistance = 12
            for _ in 0..<8 {
                let player = AVAudioPlayerNode()
                engine.attach(player)
                engine.connect(player, to: environment, format: format)
                player.renderingAlgorithm = .equalPowerPanning
                players.append(player)
            }
            buffers = GameSound.allCases.map { Self.synthesize($0, format: format) }
            try engine.start()
            players.forEach { $0.play() }
            running = true
            Log.info("game: audio ready (\(Int(session.ioBufferDuration * 1000)) ms buffer)")
        } catch {
            Log.info("game: audio unavailable: \(error.localizedDescription)")
        }
    }

    /// A short knock: decaying partials (the ball's and the surface's resonances) plus a click of
    /// noise at the impact.
    private static func synthesize(_ kind: GameSound, format: AVAudioFormat) -> AVAudioPCMBuffer {
        // (frequency Hz, amplitude, decay s), then noise amplitude and decay.
        let partials: [(Float, Float, Float)]
        let noise: (Float, Float)
        switch kind {
        case .paddle: partials = [(1050, 1, 0.016), (2350, 0.45, 0.010), (4100, 0.2, 0.006)]; noise = (0.35, 0.0025)
        case .table: partials = [(1550, 1, 0.028), (3300, 0.4, 0.014), (680, 0.35, 0.040)]; noise = (0.3, 0.002)
        case .net: partials = [(240, 0.7, 0.05), (520, 0.3, 0.03)]; noise = (0.6, 0.02)
        case .floor: partials = [(1800, 0.8, 0.018), (900, 0.4, 0.03)]; noise = (0.3, 0.002)
        }
        let rate = Float(format.sampleRate)
        let count = AVAudioFrameCount(0.15 * format.sampleRate)
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: count)!
        buffer.frameLength = count
        let out = buffer.floatChannelData![0]
        var random = GameRandom(seed: UInt64(kind.rawValue + 1))
        var lowpass: Float = 0
        var peak: Float = 1e-6
        for i in 0..<Int(count) {
            let t = Float(i) / rate
            var s: Float = 0
            for (f, a, d) in partials { s += a * sin(2 * .pi * f * t) * exp(-t / d) }
            let white = Float(random.next() * 2 - 1)
            lowpass += (white - lowpass) * (kind == .net ? 0.15 : 0.6)
            s += noise.0 * lowpass * exp(-t / noise.1)
            s *= min(t / 0.0005, 1)                        // no click at the very start
            out[i] = s
            peak = max(peak, abs(s))
        }
        for i in 0..<Int(count) { out[i] *= 0.9 / peak }
        return buffer
    }
}
