// HaruEngine.swift - Haru in-process: one MLX thread, prepared prompts, live conversations
// Copyright 2026 Vincent Gourbin
//
// Replaces upstream's model server (haru_mlx/native.py, backends.py) inside the app: no socket, no
// Python. Everything MLX runs on `worker`; only Swift values ([Float], Frame) leave it.

import Foundation
import MLX

public enum HaruPrecision: String, Sendable, CaseIterable, Codable {
    case bf16, int8, fp32
}

public struct ConversationSettings: Sendable, Equatable {
    public var systemPrompt = ""
    /// Mono float32 at 24 kHz, already prepared (`AudioFile.prepareReference`).
    public var reference: [Float]?
    public var config = DuplexConfig()
    public init(systemPrompt: String = "", reference: [Float]? = nil, config: DuplexConfig = DuplexConfig()) {
        self.systemPrompt = systemPrompt
        self.reference = reference
        self.config = config
    }
}

extension ConversationSettings {
    /// What the prepared prompt depends on; the rest (sampling) is applied by `DuplexSession.retune`.
    func samePrompt(as other: ConversationSettings) -> Bool {
        systemPrompt == other.systemPrompt && reference == other.reference && config.speakFirst == other.config.speakFirst
            && config.seed == other.config.seed && config.referenceSeconds == other.config.referenceSeconds
            && (config.dry == nil) == (other.config.dry == nil)
    }
}

public final class HaruEngine: @unchecked Sendable {
    public let worker = MLXWorker()
    public let precision: HaruPrecision
    public let frameSamples: Int
    public let sampleRate: Int
    /// Median step at the live cadence plus 3 ms (upstream `measure_step`), set by `measureStep`.
    public private(set) var stepSeconds = 0.06
    private var model: HaruModel!
    private var prepared: (settings: ConversationSettings, session: DuplexSession, first: Frame)?

    private init(precision: HaruPrecision, frameSamples: Int, sampleRate: Int) {
        self.precision = precision
        self.frameSamples = frameSamples
        self.sampleRate = sampleRate
    }

    /// Loads the checkpoint on the engine's MLX thread (the cache limit as upstream's `LocalBackend`).
    public static func load(directory: URL = HaruWeights.directory(), precision: HaruPrecision = .bf16) async throws -> HaruEngine {
        let config = try HaruConfiguration.load(contentsOf: directory.appendingPathComponent("config.json"))
        let engine = HaruEngine(precision: precision, frameSamples: config.frameSamples, sampleRate: config.sampleRate)
        let tokenizer = try await HaruTokenizer.load(directory: directory)
        engine.model = try await engine.worker.run {
            // 1 GiB as upstream's LocalBackend; HARU_CACHE_LIMIT_MB overrides it (measurements).
            Memory.cacheLimit = (ProcessInfo.processInfo.environment["HARU_CACHE_LIMIT_MB"].flatMap(Int.init) ?? 1024) << 20
            let store = try WeightStore.load(checkpoint: directory.appendingPathComponent("model.safetensors"), config: config,
                                             dtype: precision == .fp32 ? .float32 : nil)
            let model = HaruModel(store, config: config)
            model.attach(tokenizer)
            model.evaluateParameters()
            if precision == .int8 { try model.quantize(bits: 8) }
            return model
        }
        return engine
    }

    public var tokenizer: HaruTokenizer { model.tokenizer! }

    /// A prompter for this model's tokens and frame rate.
    public func makePrompter(_ settings: Prompter.Settings = Prompter.Settings()) -> Prompter {
        Prompter(tokenizer: model.tokenizer!, config: model.config, settings: settings)
    }

    /// MLX memory (active, cache, peak) in bytes, read on the MLX thread.
    public func mlxMemory() async throws -> (active: Int, cache: Int, peak: Int) {
        try await worker.run { (Memory.activeMemory, Memory.cacheMemory, Memory.peakMemory) }
    }

    /// A throwaway conversation fed at the live cadence (blocks of 1/20 frame, CPU kept awake by polling,
    /// as the pump): the first call of every shape is slow, and the step measured here sets the playback lead.
    public func measureStep(settings: ConversationSettings, frames: Int = 12) async throws {
        stepSeconds = try await worker.run { [self] in
            let session = DuplexSession(model: model, systemPrompt: settings.systemPrompt, reference: settings.reference,
                                        config: { var c = settings.config; c.seed = 0; return c }())
            _ = session.start()
            let blocks = 20, block = [Float](repeating: 0, count: frameSamples / blocks)
            let period = Double(frameSamples) / Double(sampleRate) / Double(blocks)
            var steps: [Double] = []
            for _ in 0 ..< (frames + 8) {  // 8 warm-up frames: the GPU clock is still rising
                let start = Date()
                for i in 0 ..< blocks {
                    session.feed(block)
                    let due = start.addingTimeInterval(Double(i + 1) * period)
                    LivePump.wait(until: due, warm: 1)  // poll as the live pump does
                }
                steps.append(try session.step().seconds)
            }
            let measured = steps.suffix(frames).sorted()
            return measured[measured.count / 2] + 0.003
        }
    }

    /// Runs the prompt of `settings` in the background; `begin` with the same settings then starts at once.
    public func prepare(_ settings: ConversationSettings) async throws {
        if let prepared, prepared.settings.samePrompt(as: settings) { return }
        let result = try await worker.run { [self] () -> (DuplexSession, Frame) in
            let session = DuplexSession(model: model, systemPrompt: settings.systemPrompt, reference: settings.reference,
                                        config: settings.config)
            return (session, session.start())
        }
        prepared = (settings, result.0, result.1)
    }

    /// Starts a live conversation: model frame 0 is delivered at once, then frame k + 1 as soon as user
    /// frame k is complete. `onFrame` runs on the MLX thread: keep it short (hand off and return).
    /// `onUserLevel` (same thread) gets the level in dBFS of each complete user frame, for timelines.
    /// With a `prompter`, its lines replace Haru's text (see Prompter) and it observes every frame.
    public func begin(_ settings: ConversationSettings, prompter: Prompter? = nil,
                      onUserLevel: (@Sendable (Int, Float) -> Void)? = nil,
                      onFrame: @escaping @Sendable (Frame) -> Void) async throws -> LiveConversation {
        try await prepare(settings)
        guard let (_, session, first) = prepared else { throw HaruError.invalidRequest("nothing prepared") }
        prepared = nil
        try await worker.run {
            session.retune(settings.config)  // sampling may have changed since `prepare`
            session.textOverride = prompter.map { p in { p.override($0, $1) } }
            session.guidanceOverride = prompter.map { p in { p.guidance($0) } }
        }
        let conversation = LiveConversation(sampleRate: sampleRate, frameSamples: frameSamples)
        onFrame(first)
        worker.submit {
            LivePump(session: session, conversation: conversation, onFrame: onFrame, onUserLevel: onUserLevel,
                     prompter: prompter).run()
        }
        return conversation
    }
}

extension HaruModel {
    func attach(_ tokenizer: HaruTokenizer) { self.tokenizer = tokenizer }

    func evaluateParameters() {
        for layer in lm.layers { eval(layer.arrays) }
        eval(lm.embedTokens.arrays + [lm.norm.weight, latentScale, latentBias])
        eval(modelAudioProj.arrays + userAudioProj.arrays + head.arrays + codec.arrays)
    }
}
