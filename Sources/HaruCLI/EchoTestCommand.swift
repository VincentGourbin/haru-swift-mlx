// EchoTestCommand.swift - `haru echo-test`: how much of what is played comes back into the microphone
// Copyright 2026 Vincent Gourbin
//
// No model. Plays a voice clip through the output, as model frames, and records what would reach
// the model. Reports the microphone level while the clip plays against while it is silent, with
// and without Apple's echo canceller, and the loopback delay (played -> recorded) by correlation.

import ArgumentParser
import Foundation
import HaruAudio
import HaruCore

struct EchoTest: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "echo-test",
        abstract: "Play a clip through the speakers and measure how much returns into the microphone (sound plays!).")

    @Argument(help: "Mono clip to play (a voice, 24 kHz or any format AVFoundation reads).") var clip: String
    @Option(help: "Input device name (default: the system input).") var input: String?
    @Option(help: "Output device name (default: the system output).") var output: String?
    @Flag(help: "Use Apple's voice-processing unit (echo cancellation).") var echoCancellation = false
    @Option(help: "Silence before the clip, in seconds (the noise floor).") var lead: Double = 3
    @Option(help: "Write what the microphone delivered (24 kHz WAV).") var heard: String?

    func run() async throws {
        guard let inDevice = findDevice(input, input: true), let outDevice = findDevice(output, input: false) else {
            throw ValidationError("no such device")
        }
        let frameSamples = 3200
        var voice = try AudioFile.load(URL(fileURLWithPath: (clip as NSString).expandingTildeInPath))
        voice = [Float](repeating: 0, count: Int(lead * 24_000)) + voice + [Float](repeating: 0, count: 24_000)
        let frames = (voice.count + frameSamples - 1) / frameSamples
        voice += [Float](repeating: 0, count: frames * frameSamples - voice.count)
        let recorder = Recorder()
        let audio = DuplexAudio(input: inDevice, output: outDevice, bufferMs: 2, marginMs: 15,
                                frameSamples: frameSamples * 2, stepMs: 0, echoCancellation: echoCancellation) {
            recorder.add($0)
        }
        try audio.start()
        let start = Date()
        for k in 0 ..< frames {
            let due = start.addingTimeInterval(Double(k) * Double(frameSamples) / 24_000)
            let wait = due.timeIntervalSinceNow
            if wait > 0 { try await Task.sleep(nanoseconds: UInt64(wait * 1e9)) }
            audio.receive(index: k, pcm: upsample2x(Array(voice[(k * frameSamples) ..< ((k + 1) * frameSamples)])), step: 0)
        }
        try await Task.sleep(nanoseconds: 1_500_000_000)
        audio.stop()
        let mic48 = recorder.samples
        let mic = stride(from: 0, to: mic48.count / 2 * 2, by: 2).map { (mic48[$0] + mic48[$0 + 1]) * 0.5 }
        if let heard { try AudioFile.writeWAV(mic, to: URL(fileURLWithPath: (heard as NSString).expandingTildeInPath)) }

        // Loopback delay: the lag (0-600 ms) that best aligns the mic with the played clip's envelope.
        let hop = 240  // 10 ms
        func envelope(_ x: [Float]) -> [Float] {
            stride(from: 0, to: x.count - hop, by: hop).map { i in x[i ..< i + hop].reduce(0) { $0 + $1 * $1 } }
        }
        let played = envelope(voice), recorded = envelope(mic)
        var bestLag = 0, best = -Double.infinity
        for lag in 0 ..< 60 {
            var dot = 0.0
            for i in 0 ..< min(played.count, recorded.count - lag) { dot += Double(played[i]) * Double(recorded[i + lag]) }
            if dot > best { best = dot; bestLag = lag }
        }
        // Mic level, frame by frame, aligned on the played clip (shifted by the loopback delay).
        let shift = bestLag * hop
        var during: [Float] = [], quiet: [Float] = []
        for k in 0 ..< frames {
            let a = k * frameSamples + shift, b = a + frameSamples
            guard b <= mic.count else { break }
            let playedLevel = TurnDetector.level(voice[(k * frameSamples) ..< ((k + 1) * frameSamples)])
            let micLevel = TurnDetector.level(mic[a ..< b])
            if playedLevel > -40 { during.append(micLevel) } else if playedLevel < -80 { quiet.append(micLevel) }
        }
        func med(_ v: [Float]) -> Float { v.isEmpty ? .nan : v.sorted()[v.count / 2] }
        func p90(_ v: [Float]) -> Float { v.isEmpty ? .nan : v.sorted()[min(v.count - 1, Int(0.9 * Double(v.count)))] }
        print(String(format: "%@ · input %@ (%.0f Hz) · output %@ (%.0f Hz)", echoCancellation ? "echo cancellation ON" : "raw (no echo cancellation)",
                     inDevice.name, audio.runningInputRate, outDevice.name, audio.runningOutputRate))
        for note in audio.notes { print("  note: \(note)") }
        print(String(format: "mic level while the clip plays: median %.1f dBFS, p90 %.1f · while silent: median %.1f, p90 %.1f · echo above floor: %+.1f dB",
                     med(during), p90(during), med(quiet), p90(quiet), med(during) - med(quiet)))
        print(String(format: "loopback delay (played -> recorded, by correlation): %d ms; device latencies stated: in %.1f ms, out %.1f ms",
                     bestLag * 10, audio.inLatency * 1000, audio.outLatency * 1000))
    }
}

final class Recorder: @unchecked Sendable {
    private let lock = NSLock()
    private var buffer: [Float] = []
    init() { buffer.reserveCapacity(48_000 * 120) }
    func add(_ x: UnsafeBufferPointer<Float>) { lock.lock(); buffer.append(contentsOf: x); lock.unlock() }
    var samples: [Float] { lock.lock(); defer { lock.unlock() }; return buffer }
}
