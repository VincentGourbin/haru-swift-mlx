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
    /// The system defaults when last seen: a selection equal to them follows the system.
    private var defaultInputName = ""
    private var defaultOutputName = ""
    private var watcher: DeviceWatcher?
    /// Apple's echo canceller (voice processing): for speakers without headphones.
    var echoCancellation = UserDefaults.standard.bool(forKey: "echoCancellation") {
        didSet { UserDefaults.standard.set(echoCancellation, forKey: "echoCancellation") }
    }
    /// Writes what the model heard to the log folder at the end of a conversation (troubleshooting).
    var keepHeardAudio = UserDefaults.standard.bool(forKey: "keepHeardAudio") {
        didSet { UserDefaults.standard.set(keepHeardAudio, forKey: "keepHeardAudio") }
    }
    var guidance: Float = 1.5 { didSet { schedulePrepare() } }
    var speakFirst = false { didSet { schedulePrepare() } }
    /// Added to the "start speaking" logit: > 0 makes Haru take the turn sooner (upstream `eop_bias`). 0.5 by
    /// default: 4.5 of 5 questions answered instead of 3.9 (docs/knowledge/investigations/turn-timing-swift-vs-python.md).
    var eagerness: Float = UserDefaults.standard.object(forKey: "eagerness") as? Float ?? 0.5 {
        didSet { UserDefaults.standard.set(eagerness, forKey: "eagerness"); schedulePrepare() }
    }
    var temperature: Float = 0.7 { didSet { schedulePrepare() } }
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
    /// Per-frame levels (dBFS) of both sides and Haru's text per frame, for the timeline.
    private(set) var userLevels: [Float] = []
    private(set) var modelLevels: [Float] = []
    private var frameText: [(Int, String)] = []
    private(set) var timeline: [TimelineEntry] = []
    private var lastTimelineUpdate = Date.distantPast
    private var lastLiveLog = Date.distantPast
    private var loggedEntries = Set<String>()
    private var lastUser: [Float] = []
    private var meter: Timer?
    private var feeder: StudioFeeder?
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
        defaultInputName = inputName
        defaultOutputName = outputName
        watcher = DeviceWatcher { [weak self] in Task { @MainActor in self?.devicesChanged() } }
        loadExampleCard()
        Task { await load() }
    }

    var settings: ConversationSettings {
        var config = DuplexConfig()
        config.guidanceScale = guidance
        config.speakFirst = speakFirst
        config.eopBias = eagerness
        config.temperature = temperature
        let prompt = (try? library?.render(card, language: language)) ?? ""
        return ConversationSettings(systemPrompt: prompt, reference: reference, config: config)
    }

    /// Headphones plugged in or out, defaults changed: refresh the list; a selection that followed
    /// the system default (or whose device is gone) moves to the new default. Not during a conversation.
    private func devicesChanged() {
        devices = audioDevices()
        let newIn = devices.first { $0.id == defaultDevice(input: true) }?.name ?? ""
        let newOut = devices.first { $0.id == defaultDevice(input: false) }?.name ?? ""
        let hasIn = devices.contains { $0.name == inputName && $0.inputs > 0 }
        let hasOut = devices.contains { $0.name == outputName && $0.outputs > 0 }
        if phase != .live {
            if inputName == defaultInputName || !hasIn { inputName = newIn }
            if outputName == defaultOutputName || !hasOut { outputName = newOut }
        }
        defaultInputName = newIn
        defaultOutputName = newOut
        StudioLog.write("devices changed: input \(inputName), output \(outputName) (system: \(newIn) / \(newOut))" +
                        (phase == .live ? " — applies to the next conversation" : ""))
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
        devices = audioDevices()  // fresh rates and plugged-in devices
        guard let input = devices.first(where: { $0.name == inputName && $0.inputs > 0 }) ?? findDevice(nil, input: true),
              let output = devices.first(where: { $0.name == outputName && $0.outputs > 0 }) ?? findDevice(nil, input: false)
        else { phase = .failed("No audio device"); return }
        transcript = ""
        report = ""
        modelFrames = [:]
        userLevels = []
        modelLevels = []
        frameText = []
        timeline = []
        loggedEntries = []
        StudioLog.write("start: input \(input.name), output \(output.name), voice \(String(format: "%.0f", voiceSeconds)) s, " +
                        "guidance \(guidance), eagerness \(eagerness), temperature \(temperature), margin \(Int(marginMs)) ms, \(precision.rawValue)")
        let feeder = StudioFeeder()
        let audio = DuplexAudio(input: input, output: output, bufferMs: 2, marginMs: marginMs,
                                frameSamples: engine.frameSamples * 2, stepMs: engine.stepSeconds * 1000,
                                echoCancellation: echoCancellation) { feeder.push($0) }
        let playback = DispatchQueue(label: "haru.studio.playback", qos: .userInteractive)
        do {
            let conversation = try await engine.begin(settings, onUserLevel: { [weak self] index, level in
                Task { @MainActor in self?.receivedUser(index, level) }
            }) { [weak self] frame in
                playback.async { audio.receive(index: frame.index, pcm: upsample2x(frame.audio), step: frame.seconds) }
                Task { @MainActor in self?.received(frame) }
            }
            conversation.recordsUser = true
            feeder.target = conversation
            try audio.start()
            StudioLog.write(String(format: "devices: input %@ %.0f Hz (listed %.0f), output %@ %.0f Hz (listed %.0f), echo cancellation %@, lead %.0f ms",
                                   input.name, audio.runningInputRate, input.rate, output.name, audio.runningOutputRate, output.rate,
                                   echoCancellation ? "on" : "off", audio.lead * 1000))
            for note in audio.notes { StudioLog.write("audio: \(note)") }
            self.feeder = feeder
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
        if !frame.text.isEmpty { frameText.append((frame.index, frame.text)) }
        set(&modelLevels, frame.index, TurnDetector.level(frame.audio[...]))
        updateTimeline()
    }

    private func receivedUser(_ index: Int, _ level: Float) {
        set(&userLevels, index, level)
        updateTimeline()
    }

    private func set(_ levels: inout [Float], _ index: Int, _ level: Float) {
        if index >= levels.count { levels += [Float](repeating: -120, count: index + 1 - levels.count) }
        levels[index] = level
    }

    /// Rebuilt from the levels at most a few times a second (cheap: a few thousand frames).
    private func updateTimeline(force: Bool = false) {
        guard force || Date().timeIntervalSince(lastTimelineUpdate) > 0.25 else { return }
        lastTimelineUpdate = Date()
        timeline = TimelineEntry.build(user: userLevels, model: modelLevels, text: frameText)
    }

    private func poll() {
        guard let audio else { return }
        inputLevel = audio.inputLevel
        outputLevel = audio.outputLevel
        latencyMs = audio.recentLatency().map { $0 * 1000 }
        liveLog()
    }

    /// While live: a line every 3 s, and each stretch of speech once it is over (for watching a session).
    private func liveLog() {
        let now = Double(max(userLevels.count, modelLevels.count)) * TurnDetector().frameSeconds
        for e in timeline where e.segment.end < now - 0.8 && !loggedEntries.contains(e.id) {
            loggedEntries.insert(e.id)
            if e.speaker == .user {
                StudioLog.write(String(format: "live %.1f s  YOU   %.1f-%.1f s", now, e.segment.start, e.segment.end))
            } else {
                StudioLog.write(String(format: "live %.1f s  HARU  %.1f-%.1f s%@  %@", now, e.segment.start, e.segment.end,
                                       e.delay.map { String(format: " (reply %+.1f s)", $0) } ?? "", e.text))
            }
        }
        guard Date().timeIntervalSince(lastLiveLog) >= 3 else { return }
        lastLiveLog = Date()
        let you = userLevels.suffix(22).max() ?? -120, haru = modelLevels.suffix(22).max() ?? -120
        StudioLog.write(String(format: "live %.1f s  audio latency %@ · peak level last 3 s: you %.0f dBFS, haru %.0f dBFS",
                               now, latencyMs.map { String(format: "%.0f ms", $0) } ?? "-", you, haru))
    }

    func stop() {
        meter?.invalidate()
        audio?.stop()
        conversation?.stopAndWait()
        if let audio, let conversation {
            let r = audio.report()
            let steps = conversation.steps.dropFirst(8).sorted()
            let p99 = steps.isEmpty ? 0 : steps[min(steps.count - 1, Int(0.99 * Double(steps.count)))] * 1000
            report = String(format: "Audio latency: median %.0f ms, p95 %.0f ms · underruns %d · step p50 %.1f ms, p99 %.1f ms",
                            r.p50 * 1000, r.p95 * 1000, r.underruns,
                            steps.isEmpty ? 0 : steps[steps.count / 2] * 1000, p99)
            lastUser = conversation.userAudio
            if let feeder {
                StudioLog.write(String(format: "microphone: %.0f samples/s reached the model after resampling (expected 48000), %.1f s of user audio kept",
                                       feeder.rate, Double(lastUser.count) / 24_000))
            }
            // What the model heard, for troubleshooting (opt-in; local log folder, overwritten each time).
            let heard = StudioLog.url.deletingLastPathComponent().appendingPathComponent("last-user-heard.wav")
            if keepHeardAudio { try? AudioFile.writeWAV(lastUser, to: heard) } else { try? FileManager.default.removeItem(at: heard) }
            updateTimeline(force: true)
            var userDetector = TurnDetector()
            userDetector.thresholdDB = TurnDetector.adaptiveThreshold(userLevels)
            let turns = userDetector.turns(user: userDetector.segments(userLevels),
                                           model: TurnDetector().segments(modelLevels), text: frameText)
            let summary = turns.replySummary
            report += String(format: " · %d/%d turns answered, median reply delay %@", summary.answered, turns.count,
                             summary.median.map { String(format: "%+.2f s", $0) } ?? "—")
            StudioLog.write(report)
            StudioLog.write("audio latency parts: " + r.parts.map { String(format: "%@ %.1f ms", $0.0, $0.1 * 1000) }.joined(separator: ", ")
                            + String(format: ", in %.1f ms, out %.1f ms (devices)", audio.inLatency * 1000, audio.outLatency * 1000))
            // The "You" detection is a fixed threshold: these say whether it fits this microphone and room.
            func percentiles(_ v: [Float]) -> String {
                let s = v.filter { $0 > -119 }.sorted()
                guard !s.isEmpty else { return "-" }
                return [0.1, 0.5, 0.9, 0.99].map { String(format: "%.0f", s[min(s.count - 1, Int($0 * Double(s.count)))]) }.joined(separator: "/")
            }
            StudioLog.write("frame levels dBFS p10/p50/p90/p99: you \(percentiles(userLevels)), haru \(percentiles(modelLevels)) " +
                            "(your speech threshold \(Int(userDetector.thresholdDB)), Haru's -40)")
            // Echo: your microphone while Haru speaks against while both are quiet.
            let pairs = zip(userLevels, modelLevels)
            let whileHaru = pairs.filter { $0.1 > -40 }.map(\.0).sorted(), whileQuiet = pairs.filter { $0.1 < -70 }.map(\.0).sorted()
            if !whileHaru.isEmpty, !whileQuiet.isEmpty {
                StudioLog.write(String(format: "echo: your mic at %.0f dBFS (median) while Haru speaks, %.0f while he is silent (%+.0f dB)",
                                       whileHaru[whileHaru.count / 2], whileQuiet[whileQuiet.count / 2],
                                       whileHaru[whileHaru.count / 2] - whileQuiet[whileQuiet.count / 2]))
            }
            let memory = ProcessMemory.footprint()
            StudioLog.write(String(format: "memory: app footprint %.1f GB (peak %.1f GB)", Double(memory.current) / 1e9, Double(memory.peak) / 1e9))
            if let engine {
                Task {
                    if let m = try? await engine.mlxMemory() {
                        StudioLog.write(String(format: "memory: MLX active %.1f GB, cache %.2f GB, peak %.1f GB",
                                               Double(m.active) / 1e9, Double(m.cache) / 1e9, Double(m.peak) / 1e9))
                    }
                }
            }
            for t in turns {
                StudioLog.write(String(format: "turn you %.2f-%.2f s -> %@", t.user.start, t.user.end,
                                       t.reply.map { String(format: "haru %.2f-%.2f s (%+.2f s) %@", $0.start, $0.end, t.delay!, t.text) } ?? "no reply"))
            }
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

/// One line of the timeline: a stretch of speech by one side, with Haru's words and, when it answers
/// the utterance before it, the delay since that utterance ended.
struct TimelineEntry: Identifiable, Equatable {
    enum Speaker { case user, haru }
    let speaker: Speaker
    let segment: SpeechSegment
    let text: String
    let delay: Double?
    var id: String { "\(speaker)-\(segment.start)" }

    static func build(user: [Float], model: [Float], text: [(Int, String)]) -> [TimelineEntry] {
        let detector = TurnDetector()
        var userDetector = TurnDetector()
        userDetector.thresholdDB = TurnDetector.adaptiveThreshold(user)
        let u = userDetector.segments(user), m = detector.segments(model)
        let turns = detector.turns(user: u, model: m, text: text)
        var delays: [Double: Double] = [:]  // reply start -> delay
        for t in turns { if let r = t.reply, let d = t.delay, delays[r.start] == nil { delays[r.start] = d } }
        var entries = u.map { TimelineEntry(speaker: .user, segment: $0, text: "", delay: nil) }
        for s in m {
            let said = text.filter { Double($0.0) * detector.frameSeconds >= s.start - detector.textLead && Double($0.0) * detector.frameSeconds < s.end }
                .map(\.1).joined().trimmingCharacters(in: .whitespacesAndNewlines)
            entries.append(TimelineEntry(speaker: .haru, segment: s, text: said, delay: delays[s.start]))
        }
        return entries.sorted { $0.segment.start < $1.segment.start }
    }
}

/// Microphone blocks to the conversation (audio thread).
final class StudioFeeder: @unchecked Sendable {
    var target: LiveConversation?
    private var first = 0.0, last = 0.0, count = 0
    func push(_ samples: UnsafeBufferPointer<Float>) {
        let t = ProcessInfo.processInfo.systemUptime
        if count == 0 { first = t }
        last = t
        count += samples.count
        target?.push48k(samples)
    }
    /// Samples per second delivered by the audio thread (read after the conversation).
    var rate: Double { last > first ? Double(count) / (last - first) : 0 }
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
