import AVFoundation
import CoreVideo
import Metal
import Photos

/// Records a game as a flat video: the passthrough sphere and the scene seen from between the eyes with a
/// plain perspective (`Renderer.drawRecording`), not the lens-warped side-by-side screen, plus the game's
/// sounds. H.264 + AAC in Documents/games/*.mp4 (the Files app), then copied into Photos.
///
/// The render thread asks for a video frame (`nextFrame`) at most 30 times a second, draws into it on the
/// GPU, and hands it back; appending and audio happen on the recorder's own queue.
final class GameRecorder {
    static let width = 1280, height = 720
    static let fps = 30.0

    let url: URL
    private let writer: AVAssetWriter
    private let video: AVAssetWriterInput
    private let adaptor: AVAssetWriterInputPixelBufferAdaptor
    private let audio: AVAssetWriterInput?
    private let queue = DispatchQueue(label: "game.recorder", qos: .userInitiated)
    private var textureCache: CVMetalTextureCache?
    private var sessionStart: Double?               // render thread sets it once; read on `queue` after
    private var lastFrameTime: Double?              // render thread
    private var finished = false                    // queue
    private(set) var frames = 0                     // queue

    /// `audioFormat`: what the game's sound comes in as (nil: no sound track).
    init(device: MTLDevice, audioFormat: AVAudioFormat?) throws {
        let directory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("games")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let name = DateFormatter()
        name.dateFormat = "yyyy-MM-dd HH.mm.ss"
        url = directory.appendingPathComponent("Ping pong \(name.string(from: Date())).mp4")
        writer = try AVAssetWriter(outputURL: url, fileType: .mp4)

        video = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: Self.width,
            AVVideoHeightKey: Self.height,
            AVVideoColorPropertiesKey: [
                AVVideoColorPrimariesKey: AVVideoColorPrimaries_ITU_R_709_2,
                AVVideoTransferFunctionKey: AVVideoTransferFunction_ITU_R_709_2,
                AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_709_2,
            ],
            AVVideoCompressionPropertiesKey: [
                AVVideoAverageBitRateKey: 10_000_000,
                AVVideoExpectedSourceFrameRateKey: Int(Self.fps),
                AVVideoMaxKeyFrameIntervalKey: Int(Self.fps) * 2,
            ],
        ])
        video.expectsMediaDataInRealTime = true
        adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: video, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: Self.width,
            kCVPixelBufferHeightKey as String: Self.height,
            kCVPixelBufferMetalCompatibilityKey as String: true,
            kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any](),
        ])
        writer.add(video)

        if let format = audioFormat {
            let input = AVAssetWriterInput(mediaType: .audio, outputSettings: [
                AVFormatIDKey: kAudioFormatMPEG4AAC,
                AVSampleRateKey: format.sampleRate,
                AVNumberOfChannelsKey: min(Int(format.channelCount), 2),
                AVEncoderBitRateKey: 128_000,
            ])
            input.expectsMediaDataInRealTime = true
            if writer.canAdd(input) { writer.add(input); audio = input } else { audio = nil }
        } else {
            audio = nil
        }
        guard writer.startWriting() else { throw writer.error ?? CameraError("the video writer did not start") }
        CVMetalTextureCacheCreate(nil, nil, device, nil, &textureCache)
    }

    /// Render thread: a video frame to draw this frame (display time `time`, CACurrentMediaTime seconds)
    /// into, if one is due and the encoder keeps up; call `commit` once the GPU has drawn it.
    func nextFrame(at time: Double) -> (texture: MTLTexture, commit: () -> Void)? {
        if let last = lastFrameTime, time - last < 1 / Self.fps - 0.004 { return nil }
        guard writer.status == .writing, video.isReadyForMoreMediaData, let cache = textureCache else { return nil }
        if sessionStart == nil {
            writer.startSession(atSourceTime: CMTime(seconds: time, preferredTimescale: 1_000_000))
            sessionStart = time
        }
        guard let pool = adaptor.pixelBufferPool else { return nil }
        var pixelBuffer: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(nil, pool, &pixelBuffer) == kCVReturnSuccess, let pixelBuffer else { return nil }
        var cvTexture: CVMetalTexture?
        guard CVMetalTextureCacheCreateTextureFromImage(nil, cache, pixelBuffer, nil, .bgra8Unorm, Self.width, Self.height,
                                                        0, &cvTexture) == kCVReturnSuccess,
              let cvTexture, let texture = CVMetalTextureGetTexture(cvTexture) else { return nil }
        lastFrameTime = time
        return (texture, { [self] in
            queue.async { [self] in
                withExtendedLifetime(cvTexture) {}
                guard !finished, video.isReadyForMoreMediaData else { return }
                if adaptor.append(pixelBuffer, withPresentationTime: CMTime(seconds: time, preferredTimescale: 1_000_000)) {
                    frames += 1
                }
            }
        })
    }

    /// The game's sound (any thread; the buffer is copied before this returns).
    func addAudio(_ buffer: AVAudioPCMBuffer, at time: AVAudioTime) {
        guard let audio, time.isHostTimeValid, let start = sessionStart else { return }
        let seconds = AVAudioTime.seconds(forHostTime: time.hostTime)
        guard seconds >= start, let sample = Self.sampleBuffer(buffer, at: seconds) else { return }
        queue.async { [self] in
            guard !finished, audio.isReadyForMoreMediaData else { return }
            audio.append(sample)
        }
    }

    /// Ends the file and copies it into Photos; `done` gets the file (nil if nothing was recorded) and
    /// whether Photos has it.
    func finish(_ done: @escaping (URL?, Bool) -> Void) {
        queue.async { [self] in
            guard !finished else { return }
            finished = true
            guard frames > 0 else {
                writer.cancelWriting()
                try? FileManager.default.removeItem(at: url)
                done(nil, false)
                return
            }
            video.markAsFinished()
            audio?.markAsFinished()
            writer.finishWriting { [self] in
                guard writer.status == .completed else {
                    Log.info("game recording: failed: \(writer.error?.localizedDescription ?? "?")")
                    done(nil, false)
                    return
                }
                let url = url
                PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in
                    guard status == .authorized || status == .limited else { done(url, false); return }
                    PHPhotoLibrary.shared().performChanges({
                        _ = PHAssetCreationRequest.creationRequestForAssetFromVideo(atFileURL: url)
                    }) { ok, _ in done(url, ok) }
                }
            }
        }
    }

    private static func sampleBuffer(_ buffer: AVAudioPCMBuffer, at seconds: Double) -> CMSampleBuffer? {
        var description: CMAudioFormatDescription?
        guard CMAudioFormatDescriptionCreate(allocator: nil, asbd: buffer.format.streamDescription, layoutSize: 0,
                                             layout: nil, magicCookieSize: 0, magicCookie: nil, extensions: nil,
                                             formatDescriptionOut: &description) == noErr, let description else { return nil }
        var timing = CMSampleTimingInfo(duration: CMTime(value: 1, timescale: CMTimeScale(buffer.format.sampleRate)),
                                        presentationTimeStamp: CMTime(seconds: seconds, preferredTimescale: CMTimeScale(buffer.format.sampleRate)),
                                        decodeTimeStamp: .invalid)
        var sample: CMSampleBuffer?
        guard CMSampleBufferCreate(allocator: nil, dataBuffer: nil, dataReady: false, makeDataReadyCallback: nil, refcon: nil,
                                   formatDescription: description, sampleCount: CMItemCount(buffer.frameLength),
                                   sampleTimingEntryCount: 1, sampleTimingArray: &timing, sampleSizeEntryCount: 0,
                                   sampleSizeArray: nil, sampleBufferOut: &sample) == noErr, let sample else { return nil }
        guard CMSampleBufferSetDataBufferFromAudioBufferList(sample, blockBufferAllocator: nil, blockBufferMemoryAllocator: nil,
                                                             flags: 0, bufferList: buffer.audioBufferList) == noErr else { return nil }
        return sample
    }
}
