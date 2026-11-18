// DuplexSession.swift - a live full-duplex conversation: one user frame in, one model frame out
// Copyright 2026 Vincent Gourbin
//
// Port of haru_mlx/session.py, frames.py and prompt.py (`build_prompt`, `silence_latent`,
// `_room_tone`) (higurashi-ai/Haru @ 7fe6b76, Apache-2.0). Frame t of the model comes from the hidden
// state at the previous position; when user frame t arrives, the position text[t] + model_audio[t]
// + user_audio[t] runs and yields frame t + 1. Every call must come from the thread that loaded the model.

import Foundation
import MLX

public struct DuplexConfig: Sendable, Equatable {
    public var guidanceScale: Float = 1.5      // audio guidance, 1 to 2
    public var temperature: Float = 0.7
    public var topK = 25
    public var topP: Float = 1.0
    public var dry: DryConfiguration? = DryConfiguration()
    public var eopBias: Float = 0                // > 0 makes the model start speaking sooner
    public var speakFirst = false                // the model opens the conversation
    public var maxFrames = 27_000                // 60 minutes
    public var referenceSeconds: Double = 180
    public var seed: UInt64?
    public init() {}
}

public struct Frame: Sendable {
    public let audio: [Float]   // 3200 samples at 24 kHz
    public let token: Int
    public let text: String     // new text of this frame, "" if none
    public let index: Int
    public let seconds: Double  // compute time
    public let done: Bool
}

public final class DuplexSession {
    static let prefillChunk = 512

    public let model: HaruModel
    public private(set) var config: DuplexConfig
    public let systemPrompt: String
    let reference: [Float]?
    public private(set) var index = 0
    public private(set) var promptLength = 0
    public private(set) var noise: NoiseSource
    var cache: BackboneCache
    let codec: CodecStream
    private var token: MLXArray?
    private var latent: MLXArray?
    private var forceText = 0
    private let dry: DryState?
    private let detokenizer: StreamDetokenizer
    private let streamIds: MLXArray
    private let banned: [Int]
    /// Forced replay (parity tests): the token and latent of frame `index` instead of the sampled ones.
    var forced: ((Int) -> (Int, MLXArray))?
    /// Per frame (parity tests): the final hidden state and the head's own latent, before any forcing.
    var onTrace: ((MLXArray, MLXArray) -> Void)?
    /// Per step (parity tests): the user latent the step consumed.
    var onUserLatent: ((MLXArray) -> Void)?

    public init(model: HaruModel, systemPrompt: String = "", reference: [Float]? = nil, config: DuplexConfig = DuplexConfig()) {
        self.model = model
        self.config = config
        self.systemPrompt = systemPrompt
        let cfg = model.config
        let silent: Set<Int> = [cfg.padTokenId, cfg.eopTokenId]
        streamIds = MLXArray([Int32(cfg.padTokenId), Int32(cfg.eopTokenId)]).reshaped([1, 2])
        banned = (silent.union(cfg.lm.eosTokenId.map { [$0] } ?? [])).sorted()
        if let reference, !reference.isEmpty {
            self.reference = Array(reference.suffix(Int(config.referenceSeconds * Double(cfg.sampleRate))))
        } else {
            self.reference = nil
        }
        noise = NoiseSource(seed: 0)
        cache = model.newCache()
        codec = CodecStream(model.codec)
        let exempt = Set(banned)
        dry = config.dry.map { DryState($0, exempt: exempt) }
        detokenizer = StreamDetokenizer(model.tokenizer!, silent: silent)
    }

    /// Sampling settings that do not touch the prompt (temperature, top-k/p, EOP bias, guidance,
    /// length), applied to the frames after this call: the app changes them without re-running the prompt.
    public func retune(_ other: DuplexConfig) {
        config.temperature = other.temperature
        config.topK = other.topK
        config.topP = other.topP
        config.eopBias = other.eopBias
        config.guidanceScale = other.guidanceScale
        config.maxFrames = other.maxFrames
    }

    // MARK: - Prompt

    /// Runs the prompt (system prompt + voice reference) and produces frame 0.
    public func start() -> Frame {
        let started = Date()
        noise = NoiseSource(seed: config.seed ?? NoiseSource.randomSeed())
        codec.reset()
        cache = model.newCache()
        let prompt = buildPrompt(key: noise.nextKey())
        forceText = config.speakFirst ? 1 : 0
        let n = prompt.tokens.dim(0)
        var hidden: MLXArray!
        for s in stride(from: 0, to: n, by: Self.prefillChunk) {
            let e = min(n, s + Self.prefillChunk)
            let emb = model.embed(tokens: prompt.tokens[s ..< e].expandedDimensions(axis: 0),
                                  modelLatents: prompt.modelLatents[s ..< e].expandedDimensions(axis: 0),
                                  userLatents: prompt.userLatents[s ..< e].expandedDimensions(axis: 0),
                                  modelMask: prompt.modelMask[s ..< e].expandedDimensions(axis: 0),
                                  userMask: prompt.userMask[s ..< e].expandedDimensions(axis: 0))
            hidden = model.forward(emb, cache)[0..., -1]
            eval([hidden] + cache.arrays)
        }
        promptLength = n
        index = 0
        // Encoding the reference and running the prompt leave large freed buffers in MLX's pool.
        Memory.clearCache()
        return emit(hidden, started: started)
    }

    struct Prompt { let tokens, modelLatents, userLatents, modelMask, userMask: MLXArray }

    func buildPrompt(key: MLXArray) -> Prompt {
        let cfg = model.config, pad = cfg.padTokenId, eop = cfg.eopTokenId, dim = cfg.latentDim
        let card = systemPrompt.isEmpty ? [] : model.tokenizer!.systemPromptIds(systemPrompt)
        var refLatents: MLXArray?
        if let reference {
            refLatents = model.codec.sample(model.codec.encode(MLXArray(reference)), key: key)[0]
        }
        let nRef = refLatents?.dim(0) ?? 0
        // A reference already ends in EOP, so an opening turn starts with PAD, EOP.
        var grid = config.speakFirst ? [pad, eop] : []
        if card.isEmpty && nRef == 0 && grid.isEmpty { grid = [pad] }
        let tokens = card + (nRef > 0 ? Array(repeating: pad, count: nRef - 1) + [eop] : []) + grid
        let n = tokens.count, nCard = card.count, nGrid = grid.count
        var modelParts = [MLXArray.zeros([nCard, dim])]
        var userParts = [MLXArray.zeros([n - nGrid, dim])]
        if let refLatents { modelParts.append(refLatents) }
        if nGrid > 0 {
            modelParts.append(broadcast(roomTone(refLatents), to: [nGrid, dim]))
            userParts.append(broadcast(silenceLatent(), to: [nGrid, dim]))
        }
        let positions = MLXArray(0 ..< n)
        return Prompt(tokens: MLXArray(tokens.map(Int32.init)), modelLatents: concatenated(modelParts),
                      userLatents: concatenated(userParts), modelMask: positions .>= nCard,
                      userMask: positions .>= (n - nGrid))
    }

    /// Latent of digital silence: the input of an audio stream with no sound yet.
    func silenceLatent() -> MLXArray {
        model.codec.encode(MLXArray.zeros([16 * model.config.codec.frameSamples]))[0, -1]
    }

    /// The reference's quietest frame (the speaker's noise floor), else silence.
    func roomTone(_ refLatents: MLXArray?) -> MLXArray {
        let frame = model.config.codec.frameSamples
        guard let reference, let refLatents, reference.count / frame > 0 else { return silenceLatent() }
        let whole = reference.count / frame
        var best = 0, bestEnergy = Float.infinity
        for f in 0..<whole {
            var e: Float = 0
            for i in (f * frame) ..< ((f + 1) * frame) { e += reference[i] * reference[i] }
            if e / Float(frame) < bestEnergy { bestEnergy = e / Float(frame); best = f }
        }
        return refLatents[best]
    }

    // MARK: - Frames

    /// Takes user audio of any length (float32, 24 kHz) as it arrives.
    public func feed(_ audio: [Float]) {
        codec.feed(MLXArray(audio))
        eval(codec.arrays)
    }

    /// User frames fed and not yet consumed by `step()`.
    public var framesReady: Int { codec.framesReady }

    /// Consumes one user frame (3200 samples) — or the one already fed — and returns the next model frame.
    public func step(_ userAudio: [Float]? = nil) throws -> Frame {
        guard let token, let latent else { throw HaruError.invalidRequest("call start() first") }
        let started = Date()
        if let userAudio {
            guard userAudio.count == model.config.frameSamples else {
                throw HaruError.invalidRequest("a user frame is \(model.config.frameSamples) samples, got \(userAudio.count)")
            }
            codec.feed(MLXArray(userAudio))
        }
        let userLatent = model.codec.sample(codec.pop(), key: noise.nextKey())
        onUserLatent?(userLatent)
        let emb = model.embed(tokens: token.reshaped([1, 1]), modelLatents: latent.reshaped([1, 1, -1]), userLatents: userLatent)
        let hidden = model.forward(emb, cache)[0..., -1]
        return emit(hidden, started: started)
    }

    /// Samples the next frame's text token and audio from `hidden [1, H]`.
    func emit(_ hidden: MLXArray, started: Date) -> Frame {
        var logits = model.logits(hidden)
        if let dry { logits = dry.apply(logits) }
        logits = putAlong(logits, streamIds, values: model.logitsFP32(hidden, tokens: streamIds[0]), axis: -1)
        if config.eopBias != 0 {
            let eop = MLXArray([Int32(model.config.eopTokenId)]).reshaped([1, 1])
            logits = putAlong(logits, eop, values: takeAlong(logits, eop, axis: -1) + config.eopBias, axis: -1)
        }
        if forceText > 0 {
            let ids = MLXArray(banned.map(Int32.init)).reshaped([1, -1])
            logits = putAlong(logits, ids, values: MLXArray.full([1, banned.count], values: MLXArray(-Float.infinity)), axis: -1)
            forceText -= 1
        }
        var nextToken = sampleToken(logits, temperature: config.temperature, topK: config.topK, topP: config.topP,
                                    key: noise.nextKey())
        let headNoise = noise.normal([1, model.config.latentDim], dtype: model.dtype)
        var nextLatent = model.unscaleLatents(model.head.sample(hidden, noise: headNoise, guidance: config.guidanceScale))
        onTrace?(hidden, nextLatent)
        if let forced {
            let (t, l) = forced(index)
            nextToken = MLXArray([Int32(t)]).reshaped(nextToken.shape)
            nextLatent = l.asType(nextLatent.dtype).reshaped(nextLatent.shape)
        }
        let audio = clip(codec.decode(nextLatent.expandedDimensions(axis: 0))[0], min: -1, max: 1)
        eval([nextToken, nextLatent, audio] + cache.arrays + codec.arrays)

        token = nextToken
        latent = nextLatent
        let id = nextToken.item(Int.self)
        dry?.update(id)
        let frame = Frame(audio: audio.asArray(Float.self), token: id, text: detokenizer.push(id), index: index,
                          seconds: Date().timeIntervalSince(started), done: index + 1 >= config.maxFrames)
        index += 1
        return frame
    }
}
