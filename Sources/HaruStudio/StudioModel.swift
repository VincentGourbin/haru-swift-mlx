// StudioModel.swift - state of Haru Studio: model, card, voice, devices, live conversation
// Copyright 2026 Vincent Gourbin

import AVFoundation
import Foundation
import HaruAudio
import HaruCore
import Observation

@MainActor @Observable
final class StudioModel {
    enum Phase: Equatable {
        case loading(String), idle, preparing, ready, live, failed(String)
    }

    // Model
    var phase: Phase = .loading("Loading Haru-4B…") { didSet { StudioLog.write("phase \(phase)") } }
    var precision: HaruPrecision = .bf16
    private(set) var engine: HaruEngine?
    private(set) var stepMs = 0.0

    // Card
    let library = try? CardLibrary.bundled()
    var language = "en" { didSet { if language != oldValue { loadExampleCard() } } }
    var card = CharacterCard() { didSet { schedulePrepare() } }

    // Voice
    var voiceURL: URL? { didSet { loadVoice() } }
    private(set) var voiceSeconds = 0.0
    private var reference: [Float]?

    // Settings
    var devices: [Device] = audioDevices()
    var inputName = ""
    var outputName = ""
    var guidance: Float = 1.5 { didSet { schedulePrepare() } }
    var speakFirst = false { didSet { schedulePrepare() } }
    var marginMs = 15.0

    // Live
    var transcript = ""
    var inputLevel: Float = 0
    var outputLevel: Float = 0
    var latencyMs: Double?
    var report = ""
    private var audio: DuplexAudio?
    private var conversation: LiveConversation?
    private var modelFrames: [Int: [Float]] = [:]
    private var lastUser: [Float] = []
    private var meter: Timer?
    private var prepareTask: Task<Void, Never>?

    init() {
        if let path = UserDefaults.standard.string(forKey: "voicePath") ?? ProcessInfo.processInfo.environment["HARU_TEST_VOICE"] {
            if FileManager.default.isReadableFile(atPath: path) {
                voiceURL = URL(fileURLWithPath: path)  // didSet loads it (@Observable runs observers in init)
            } else {
                // Downloads, Desktop and Documents need the user's consent (TCC): choosing the file grants it.
                StudioLog.write("voice \(URL(fileURLWithPath: path).lastPathComponent) not readable: choose it with Choose…")
            }
        }
        inputName = devices.first { $0.id == defaultDevice(input: true) }?.name ?? ""
        outputName = devices.first { $0.id == defaultDevice(input: false) }?.name ?? ""
        loadExampleCard()
        Task { await load() }
    }

    var settings: ConversationSettings {
        var config = DuplexConfig()
        config.guidanceScale = guidance
        config.speakFirst = speakFirst
        let prompt = (try? library?.render(card, language: language)) ?? ""
        return ConversationSettings(systemPrompt: prompt, reference: reference, config: config)
    }

    // MARK: - Model

    func load() async {
        phase = .loading("Loading Haru-4B (\(precision.rawValue))…")
        engine = nil
        do {
            let engine = try await HaruEngine.load(precision: precision)
            phase = .loading("Measuring a frame at the live cadence…")
            try await engine.measureStep(settings: settings)
            self.engine = engine
            stepMs = engine.stepSeconds * 1000
            StudioLog.write(String(format: "loaded %@, live step %.1f ms", precision.rawValue, stepMs))
            phase = .idle
            schedulePrepare(after: 0)
        } catch {
            phase = .failed("Could not load the model: \(error.localizedDescription)")
        }
    }

    private func loadExampleCard() {
        guard let preset = (try? library?.presets(language: language))?.sorted(by: { $0.key < $1.key }).first?.value else { return }
        card = preset
    }

    private func loadVoice() {
        guard let voiceURL else { reference = nil; voiceSeconds = 0; return }
        UserDefaults.standard.set(voiceURL.path, forKey: "voicePath")
        Task.detached(priority: .userInitiated) {
            let started = Date()
            var clip: [Float]?
            do {
                clip = AudioFile.prepareReference(try AudioFile.load(voiceURL))
                StudioLog.write(String(format: "voice %@: %.1f s kept, decoded in %.2f s", voiceURL.lastPathComponent,
                                       Double(clip!.count) / 24_000, Date().timeIntervalSince(started)))
            } catch {
                StudioLog.write("voice \(voiceURL.lastPathComponent) failed: \(error)")
            }
            await MainActor.run {
                self.reference = clip
                self.voiceSeconds = Double(clip?.count ?? 0) / 24_000
                self.schedulePrepare()
            }
        }
    }

    /// A few seconds after the last edit, the prompt is read into the model in the background (as upstream).
    private func schedulePrepare(after seconds: Double = 1.5) {
        guard engine != nil, phase != .live else { return }
        prepareTask?.cancel()
        let settings = settings
        prepareTask = Task {
            try? await Task.sleep(nanoseconds: UInt64(seconds * 1e9))
            guard !Task.isCancelled, let engine else { return }
            phase = .preparing
            do {
                let started = Date()
                try await engine.prepare(settings)
                StudioLog.write(String(format: "prepared: card %d chars, voice %.1f s, in %.2f s", settings.systemPrompt.count,
                                       Double(settings.reference?.count ?? 0) / 24_000, Date().timeIntervalSince(started)))
                if !Task.isCancelled, phase == .preparing { phase = .ready }
            } catch {
                phase = .failed("Prompt failed: \(error.localizedDescription)")
            }
        }
    }

    // MARK: - Conversation

    func start() async {
        guard let engine else { return }
        // A preparation in flight would queue a second prompt run ahead of the conversation: let the
        // debounce end (cancel) or the run finish, then `begin` reuses it when the settings match.
        prepareTask?.cancel()
        await prepareTask?.value
        guard await AVCaptureDevice.requestAccess(for: .audio) else {
            phase = .failed("Microphone access denied (System Settings › Privacy & Security › Microphone).")
            return
        }
        guard let input = devices.first(where: { $0.name == inputName && $0.inputs > 0 }) ?? findDevice(nil, input: true),
              let output = devices.first(where: { $0.name == outputName && $0.outputs > 0 }) ?? findDevice(nil, input: false)
        else { phase = .failed("No audio device"); return }
        transcript = ""
        report = ""
        modelFrames = [:]
        let feeder = StudioFeeder()
        let audio = DuplexAudio(input: input, output: output, bufferMs: 2, marginMs: marginMs,
                                frameSamples: engine.frameSamples * 2, stepMs: engine.stepSeconds * 1000) { feeder.push($0) }
        let playback = DispatchQueue(label: "haru.studio.playback", qos: .userInteractive)
        do {
            let conversation = try await engine.begin(settings) { [weak self] frame in
                playback.async { audio.receive(index: frame.index, pcm: upsample2x(frame.audio), step: frame.seconds) }
                Task { @MainActor in self?.received(frame) }
            }
            conversation.recordsUser = true
            feeder.target = conversation
            try audio.start()
            self.audio = audio
            self.conversation = conversation
            phase = .live
            meter = Timer.scheduledTimer(withTimeInterval: 1.0 / 30, repeats: true) { [weak self] _ in
                Task { @MainActor in self?.poll() }
            }
        } catch {
            phase = .failed("Could not start: \(error.localizedDescription)")
        }
    }

    private func received(_ frame: Frame) {
        transcript += frame.text
        modelFrames[frame.index] = frame.audio
    }

    private func poll() {
        guard let audio else { return }
        inputLevel = audio.inputLevel
        outputLevel = audio.outputLevel
        latencyMs = audio.recentLatency().map { $0 * 1000 }
    }

    func stop() {
        meter?.invalidate()
        audio?.stop()
        conversation?.stopAndWait()
        if let audio, let conversation {
            let r = audio.report()
            let steps = conversation.steps.dropFirst(8).sorted()
            let p99 = steps.isEmpty ? 0 : steps[min(steps.count - 1, Int(0.99 * Double(steps.count)))] * 1000
            report = String(format: "Reply latency: median %.0f ms, p95 %.0f ms · underruns %d · step p50 %.1f ms, p99 %.1f ms",
                            r.p50 * 1000, r.p95 * 1000, r.underruns,
                            steps.isEmpty ? 0 : steps[steps.count / 2] * 1000, p99)
            lastUser = conversation.userAudio
            StudioLog.write(report)
        }
        audio = nil
        conversation = nil
        inputLevel = 0
        outputLevel = 0
        phase = .idle
        schedulePrepare(after: 0)
    }

    var canSave: Bool { !modelFrames.isEmpty && phase != .live }

    /// Stereo WAV at 24 kHz: the user on the left, the model on the right, frame by frame.
    func save(to url: URL) throws {
        let model = modelFrames.keys.sorted().flatMap { modelFrames[$0]! }
        let n = min(lastUser.count, model.count)
        try AudioFile.writeWAV((0 ..< n).flatMap { [lastUser[$0], model[$0]] }, to: url, channels: 2)
    }
}

/// Microphone blocks to the conversation (audio thread).
final class StudioFeeder: @unchecked Sendable {
    var target: LiveConversation?
    func push(_ samples: UnsafeBufferPointer<Float>) { target?.push48k(samples) }
}

func upsample2x(_ x: [Float]) -> [Float] {
    guard let last = x.last else { return [] }
    var up = [Float](repeating: 0, count: 2 * x.count)
    for i in 0 ..< x.count {
        up[2 * i] = x[i]
        up[2 * i + 1] = i + 1 < x.count ? (x[i] + x[i + 1]) * 0.5 : last
    }
    return up
}

/// `~/Library/Logs/Haru/studio.log`: phases and conversation reports, for troubleshooting.
enum StudioLog {
    static let url: URL = {
        let dir = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Logs/Haru")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("studio.log")
    }()

    static func write(_ line: String) {
        let text = "\(ISO8601DateFormatter().string(from: Date())) \(line)\n"
        if let handle = try? FileHandle(forWritingTo: url) {
            handle.seekToEndOfFile()
            handle.write(Data(text.utf8))
            try? handle.close()
        } else {
            try? text.write(to: url, atomically: true, encoding: .utf8)
        }
    }
}
