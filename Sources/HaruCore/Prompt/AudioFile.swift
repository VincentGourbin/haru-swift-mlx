// AudioFile.swift - audio files <-> mono float32 at 24 kHz (everything inside Haru)
// Copyright 2026 Vincent Gourbin
//
// `load` decodes any format AVFoundation reads (WAV, FLAC, MP3, AAC…) and resamples with
// AVAudioConverter, fed block by block (a single-shot conversion returned silence when upsampling:
// pitfall 24). `trimSilence` and `prepareReference` port haru_mlx/audio.py.

import AVFoundation
import Foundation

public enum AudioFile {
    public static let sampleRate = 24_000

    /// Mono float32 at `sampleRate` (channels averaged, as upstream).
    public static func load(_ url: URL, sampleRate: Int = AudioFile.sampleRate) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        let source = file.processingFormat
        guard let mono = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: source.sampleRate,
                                       channels: 1, interleaved: false),
              let target = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(sampleRate),
                                         channels: 1, interleaved: false),
              let buffer = AVAudioPCMBuffer(pcmFormat: source, frameCapacity: AVAudioFrameCount(file.length))
        else { throw HaruError.invalidRequest("unsupported audio format: \(url.lastPathComponent)") }
        try file.read(into: buffer)
        // Average the channels first, as upstream.
        let frames = Int(buffer.frameLength), channels = Int(source.channelCount)
        var averaged = [Float](repeating: 0, count: frames)
        for c in 0..<channels {
            let data = buffer.floatChannelData![c]
            for i in 0..<frames { averaged[i] += data[i] / Float(channels) }
        }
        if Int(source.sampleRate) == sampleRate { return averaged }
        return try resample(averaged, from: mono, to: target)
    }

    static func resample(_ samples: [Float], from source: AVAudioFormat, to target: AVAudioFormat) throws -> [Float] {
        guard let converter = AVAudioConverter(from: source, to: target) else {
            throw HaruError.invalidRequest("no converter \(source.sampleRate) -> \(target.sampleRate) Hz")
        }
        converter.sampleRateConverterQuality = AVAudioQuality.max.rawValue
        let block = 4096
        var position = 0
        var output: [Float] = []
        output.reserveCapacity(Int(Double(samples.count) * target.sampleRate / source.sampleRate) + block)
        let outBuffer = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: AVAudioFrameCount(block * 4))!
        while true {
            outBuffer.frameLength = 0
            var error: NSError?
            let status = converter.convert(to: outBuffer, error: &error) { _, inputStatus in
                let count = min(block, samples.count - position)
                guard count > 0 else {
                    inputStatus.pointee = .endOfStream
                    return nil
                }
                let input = AVAudioPCMBuffer(pcmFormat: source, frameCapacity: AVAudioFrameCount(count))!
                input.frameLength = AVAudioFrameCount(count)
                samples.withUnsafeBufferPointer { src in
                    input.floatChannelData![0].update(from: src.baseAddress! + position, count: count)
                }
                position += count
                inputStatus.pointee = .haveData
                return input
            }
            if let error { throw error }
            let n = Int(outBuffer.frameLength)
            if n > 0 { output.append(contentsOf: UnsafeBufferPointer(start: outBuffer.floatChannelData![0], count: n)) }
            if status == .endOfStream || (status == .inputRanDry && n == 0) { break }
        }
        return output
    }

    /// Cuts leading and trailing audio quieter than -45 dBFS (20 ms RMS windows, 0.2 s margin).
    public static func trimSilence(_ audio: [Float], sampleRate: Int = AudioFile.sampleRate) -> [Float] {
        let window = sampleRate / 50
        let count = audio.count / window
        guard count > 0 else { return audio }
        var voiced: [Int] = []
        for w in 0..<count {
            var sum: Float = 0
            for i in (w * window) ..< ((w + 1) * window) { sum += audio[i] * audio[i] }
            let rms = (sum / Float(window) + 1e-12).squareRoot()
            if 20 * log10(rms) > -45 { voiced.append(w) }
        }
        guard let first = voiced.first, let last = voiced.last else { return [] }
        let margin = Int(0.2 * Double(sampleRate))
        let start = max(0, first * window - margin)
        let end = min(audio.count, (last + 1) * window + margin)
        return Array(audio[start ..< end])
    }

    /// Voice reference clip: silence trimmed, at most the last `maxSeconds`.
    public static func prepareReference(_ audio: [Float], maxSeconds: Double = 180,
                                        sampleRate: Int = AudioFile.sampleRate) -> [Float] {
        let trimmed = trimSilence(audio, sampleRate: sampleRate)
        return Array(trimmed.suffix(Int(maxSeconds * Double(sampleRate))))
    }

    /// 16-bit PCM WAV, `channels` interleaved.
    public static func writeWAV(_ samples: [Float], to url: URL, sampleRate: Int = AudioFile.sampleRate, channels: Int = 1) throws {
        var data = Data()
        func append<T: FixedWidthInteger>(_ v: T) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
        let bytes = samples.count * 2
        data.append(contentsOf: Array("RIFF".utf8)); append(UInt32(36 + bytes))
        data.append(contentsOf: Array("WAVEfmt ".utf8)); append(UInt32(16)); append(UInt16(1)); append(UInt16(channels))
        append(UInt32(sampleRate)); append(UInt32(sampleRate * channels * 2)); append(UInt16(channels * 2)); append(UInt16(16))
        data.append(contentsOf: Array("data".utf8)); append(UInt32(bytes))
        for s in samples { append(Int16(max(-1, min(1, s)) * 32767)) }
        try data.write(to: url)
    }
}
