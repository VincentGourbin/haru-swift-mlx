// TalkCommand.swift - `haru talk`: a live conversation with Haru on this Mac's microphone and speakers
// Copyright 2026 Vincent Gourbin
// Mirrors upstream's command-line client (haru_mlx/swift/CLI/main.swift), with the model in-process.

import ArgumentParser
import Foundation
import HaruAudio
import HaruCore
import MLX

struct Devices: ParsableCommand {
    static let configuration = CommandConfiguration(abstract: "List the audio devices.")
    func run() {
        for d in audioDevices() {
            print(String(format: "%@  in %d  out %d  %.0f Hz", d.name, d.inputs, d.outputs, d.rate))
        }
    }
}

struct Talk: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Talk with Haru (wired headphones recommended). Ctrl-C or --seconds ends the conversation.")

    @OptionGroup var modelOptions: ModelOptions
    @OptionGroup var card: CardOptions
    @Option(help: "Input device name (default: the system input).") var input: String?
    @Option(help: "Output device name (default: the system output).") var output: String?
    @Option(help: "Device buffer in milliseconds.") var bufferMs: Double = 2
    @Option(help: "Playback margin in milliseconds (slack for a late step).") var marginMs: Double = 15
    @Option(help: "Audio guidance scale (1-2).") var guidanceScale: Float = 1.5
    @Flag(help: "The model opens the conversation.") var speakFirst = false
    @Option(help: "Stop after this many seconds (default: until Ctrl-C).") var seconds: Double?

    func run() async throws {
        setvbuf(stdout, nil, _IONBF, 0)  // the live transcript and the status lines in order
        guard let inDevice = findDevice(input, input: true), let outDevice = findDevice(output, input: false) else {
            throw HaruError.invalidRequest("audio device not found (haru devices lists them)")
        }
        print("loading Haru-4B (\(modelOptions.precision.rawValue))…")
        let precision: HaruPrecision = switch modelOptions.precision { case .bf16: .bf16; case .fp32: .fp32; case .int8: .int8 }
        let engine = try await HaruEngine.load(precision: precision)
        var config = DuplexConfig()
        config.guidanceScale = guidanceScale
        config.speakFirst = speakFirst
        let settings = ConversationSettings(systemPrompt: try card.systemPromptText(), reference: try card.referenceAudio(), config: config)
        try await engine.measureStep(settings: settings)
        try await engine.prepare(settings)
        print(String(format: "ready: step %.1f ms at the live cadence; in: %@, out: %@", engine.stepSeconds * 1000, inDevice.name, outDevice.name))

        let feeder = Feeder()
        let audio = DuplexAudio(input: inDevice, output: outDevice, bufferMs: bufferMs, marginMs: marginMs,
                                frameSamples: engine.frameSamples * 2, stepMs: engine.stepSeconds * 1000) { feeder.push($0) }
        let playback = DispatchQueue(label: "haru.playback", qos: .userInteractive)
        let conversation = try await engine.begin(settings) { frame in
            if !frame.text.isEmpty { print(frame.text, terminator: "") }
            playback.async { audio.receive(index: frame.index, pcm: upsample2x(frame.audio), step: frame.seconds) }
        }
        feeder.target = conversation
        try audio.start()
        print("talk now…\n")
        let stop = DispatchSemaphore(value: 0)
        signal(SIGINT, SIG_IGN)
        let sigint = DispatchSource.makeSignalSource(signal: SIGINT, queue: .main)
        sigint.setEventHandler { stop.signal() }
        sigint.resume()
        if let seconds { DispatchQueue.global().asyncAfter(deadline: .now() + seconds) { stop.signal() } }
        await withCheckedContinuation { c in DispatchQueue.global().async { stop.wait(); c.resume() } }
        audio.stop()
        conversation.stopAndWait()
        let r = audio.report()
        let steps = conversation.steps.dropFirst(8).map { $0 * 1000 }.sorted()
        func pct(_ p: Double) -> Double { steps.isEmpty ? 0 : steps[min(steps.count - 1, Int(p * Double(steps.count)))] }
        print(String(format: "\n\nreply latency: median %.0f ms, p95 %.0f ms (%d frames); underruns %d, caught up %.0f ms",
                     r.p50 * 1000, r.p95 * 1000, r.frames, r.underruns, r.caughtUpMs))
        for (name, value) in r.parts { print(String(format: "  %@ %.1f ms", name, value * 1000)) }
        print(String(format: "step: p50 %.1f ms, p99 %.1f ms, max %.1f ms over %d frames; dropped input %d samples",
                     pct(0.5), pct(0.99), steps.last ?? 0, steps.count, conversation.droppedSamples))
    }
}

/// Forwards microphone blocks to the conversation once it exists (audio thread).
final class Feeder: @unchecked Sendable {
    var target: LiveConversation?
    func push(_ samples: UnsafeBufferPointer<Float>) { target?.push48k(samples) }
}

/// 24 kHz -> 48 kHz by linear interpolation (upstream `upsample_2x`).
func upsample2x(_ x: [Float]) -> [Float] {
    guard let last = x.last else { return [] }
    var up = [Float](repeating: 0, count: 2 * x.count)
    for i in 0 ..< x.count {
        up[2 * i] = x[i]
        up[2 * i + 1] = i + 1 < x.count ? (x[i] + x[i + 1]) * 0.5 : last
    }
    return up
}

/// `haru simulate`: the live path fed by a file on the wall clock instead of a microphone.
struct Simulate: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Replay a recording into a live conversation at the audio-clock pace (no audio devices).")

    @Argument(help: "Audio of the user.") var input: String
    @Argument(help: "The model's audio (24 kHz WAV), stereo with the user on the left.") var output: String
    @OptionGroup var modelOptions: ModelOptions
    @OptionGroup var card: CardOptions
    @Option(help: "Samples per block at 48 kHz (128 = 2.7 ms, a small device buffer).") var block = 128
    @Option(help: "Write every step time (ms) as a JSON array.") var stepsJson: String?

    func run() async throws {
        let precision: HaruPrecision = switch modelOptions.precision { case .bf16: .bf16; case .fp32: .fp32; case .int8: .int8 }
        let engine = try await HaruEngine.load(precision: precision)
        let settings = ConversationSettings(systemPrompt: try card.systemPromptText(), reference: try card.referenceAudio(),
                                            config: { var c = DuplexConfig(); c.seed = 0; return c }())
        try await engine.measureStep(settings: settings)
        try await engine.prepare(settings)
        let user24 = try AudioFile.load(URL(fileURLWithPath: (input as NSString).expandingTildeInPath))
        let user48 = upsample2x(user24)
        let frameSeconds = Double(engine.frameSamples) / Double(engine.sampleRate)
        let collector = Collector()
        let start = Date().addingTimeInterval(0.05)
        let conversation = try await engine.begin(settings, onUserLevel: { collector.addLevel($0, $1) }) { frame in
            collector.add(frame, at: Date())
        }
        // Feed on the wall clock, from a high-priority thread, block by block.
        let feeder = Thread {
            var sent = 0
            while sent < user48.count {
                let n = min(self.block, user48.count - sent)
                let due = start.addingTimeInterval(Double(sent + n) / 48_000)
                let wait = due.timeIntervalSinceNow
                if wait > 0 { Thread.sleep(forTimeInterval: wait) }
                user48.withUnsafeBufferPointer { conversation.push48k(UnsafeBufferPointer(rebasing: $0[sent ..< sent + n])) }
                sent += n
            }
        }
        feeder.qualityOfService = .userInteractive
        feeder.start()
        let total = Double(user24.count) / 24_000
        try await Task.sleep(nanoseconds: UInt64((total + 0.5) * 1e9))
        conversation.stopAndWait()

        let frames = collector.frames.sorted { $0.0.index < $1.0.index }
        // Model frame k + 1 answers user frame k, complete at start + (k + 1) * frame.
        let delays = frames.dropFirst(9).map { f, at in
            at.timeIntervalSince(start.addingTimeInterval(Double(f.index) * frameSeconds)) * 1000
        }.sorted()
        if let stepsJson {
            try JSONSerialization.data(withJSONObject: conversation.steps.map { $0 * 1000 })
                .write(to: URL(fileURLWithPath: stepsJson))
        }
        let steps = conversation.steps.dropFirst(8).map { $0 * 1000 }.sorted()
        func pct(_ v: [Double], _ p: Double) -> Double { v.isEmpty ? 0 : v[min(v.count - 1, Int(p * Double(v.count)))] }
        let late = delays.filter { $0 > frameSeconds * 1000 }.count
        if ProcessInfo.processInfo.environment["HARU_SIM_DEBUG"] != nil {
            for (f, at) in frames {
                let d = at.timeIntervalSince(start.addingTimeInterval(Double(f.index) * frameSeconds)) * 1000
                if d > 80 { print(String(format: "  frame %d: %.1f ms after its user frame (step %.1f ms)", f.index, d, f.seconds * 1000)) }
            }
        }
        let reply = frames.flatMap { $0.0.audio }
        let n = min(user24.count, reply.count)
        try AudioFile.writeWAV((0 ..< n).flatMap { [user24[$0], reply[$0]] },
                               to: URL(fileURLWithPath: (output as NSString).expandingTildeInPath), channels: 2)
        print(String(format: "measured step (live) %.1f ms; MLX active %.2f GB, cache %.2f GB, peak %.2f GB", engine.stepSeconds * 1000,
                     Double(Memory.activeMemory) / 1e9, Double(Memory.cacheMemory) / 1e9, Double(Memory.peakMemory) / 1e9))
        print(String(format: "%d frames; step p50 %.1f / p99 %.1f / max %.1f ms; user frame complete -> model frame ready: p50 %.1f / p99 %.1f / max %.1f ms; %d later than a frame; dropped %d",
                     frames.count, pct(steps, 0.5), pct(steps, 0.99), steps.last ?? 0, pct(delays, 0.5), pct(delays, 0.99),
                     delays.last ?? 0, late, conversation.droppedSamples))
        print(frames.map { $0.0.text }.joined().trimmingCharacters(in: .whitespaces))
        // Turns from the live levels (as the app's timeline), checked against the levels of the file.
        let live = collector.levels.sorted { $0.0 < $1.0 }
        let offline = TurnDetector.levels(user24, frameSamples: engine.frameSamples)
        let gap = zip(live, offline).map { abs($0.0.1 - $0.1) }.max() ?? 0
        let detector = TurnDetector()
        let turns = detector.turns(user: detector.segments(live.map(\.1)),
                                   model: detector.segments(frames.map { TurnDetector.level($0.0.audio[...]) }),
                                   text: frames.map { ($0.0.index, $0.0.text) })
        let summary = turns.replySummary
        print(String(format: "user levels: %d live frames, largest gap to the file %.2g dB; %d/%d turns answered, median reply delay %@",
                     live.count, gap, summary.answered, turns.count, summary.median.map { String(format: "%+.2f s", $0) } ?? "—"))
    }
}

final class Collector: @unchecked Sendable {
    private let lock = NSLock()
    private var items: [(Frame, Date)] = []
    func add(_ frame: Frame, at date: Date) { lock.lock(); items.append((frame, date)); lock.unlock() }
    var frames: [(Frame, Date)] { lock.lock(); defer { lock.unlock() }; return items }
    private var userLevels: [(Int, Float)] = []
    func addLevel(_ index: Int, _ level: Float) { lock.lock(); userLevels.append((index, level)); lock.unlock() }
    var levels: [(Int, Float)] { lock.lock(); defer { lock.unlock() }; return userLevels }
}
