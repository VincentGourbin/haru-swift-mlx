// TextSampling.swift - top-k / top-p sampling, the DRY repetition penalty, stream detokenization
// Copyright 2026 Vincent Gourbin
//
// Port of haru_mlx/sampling.py (higurashi-ai/Haru @ 7fe6b76, Apache-2.0).

import Foundation
import MLX

/// `logits [B, vocab]` (float32) -> token ids `[B]`. `temperature <= 0` is greedy.
func sampleToken(_ logits: MLXArray, temperature: Float, topK: Int, topP: Float, key: MLXArray) -> MLXArray {
    if temperature <= 0 { return argMax(logits, axis: -1) }
    var logits = logits / temperature
    var indices: MLXArray?
    if topK > 0 && topK < logits.dim(-1) {
        let partitioned = argPartition(-logits, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        indices = partitioned
        logits = takeAlong(logits, partitioned, axis: -1)
    }
    if topP > 0 && topP < 1 {
        let order = argSort(-logits, axis: -1)
        logits = takeAlong(logits, order, axis: -1)
        indices = indices.map { takeAlong($0, order, axis: -1) } ?? order
        let probs = softmax(logits, axis: -1)
        let keep = (cumsum(probs, axis: -1) - probs) .< topP
        logits = MLX.where(keep, logits, MLXArray(-Float.infinity))
    }
    let choice = MLXRandom.categorical(logits, key: key)
    guard let indices else { return choice }
    return takeAlong(indices, expandedDimensions(choice, axis: -1), axis: -1)[.ellipsis, 0]
}

/// DRY penalty: discourage extending a token sequence that already occurred.
public struct DryConfiguration: Sendable, Equatable {
    public var multiplier: Float = 0.8
    public var base: Float = 1.75
    public var allowedLength = 1
    public var lookback = 1024
    public var maxMatchLength = 32
    public var maxPenalty: Float = 32
    public init() {}
    var historySize: Int { lookback + maxMatchLength + 1 }
}

/// `[token: penalty]` for tokens that would extend an earlier repeat (`dry_penalties`).
func dryPenalties(_ history: [Int], _ config: DryConfiguration) -> [Int: Float] {
    let n = history.count
    guard n >= max(1, config.allowedLength) else { return [:] }
    var longest: [Int: Int] = [:]
    let last = history[n - 1]
    var j = n - 2
    let stop = max(0, n - 1 - config.lookback)
    while j >= stop {
        defer { j -= 1 }
        guard history[j] == last else { continue }
        var length = 1
        while length < config.maxMatchLength && j - length >= 0 && history[j - length] == history[n - 1 - length] {
            length += 1
        }
        if length >= config.allowedLength {
            let candidate = history[j + 1]
            if length > longest[candidate, default: 0] { longest[candidate] = length }
        }
    }
    return longest.mapValues {
        min(config.maxPenalty, config.multiplier * pow(config.base, Float($0 - config.allowedLength)))
    }
}

/// Running DRY history of one text stream; `exempt` tokens never count.
final class DryState {
    let config: DryConfiguration
    let exempt: Set<Int>
    private(set) var history: [Int] = []
    private(set) var penalties: [Int: Float] = [:]

    init(_ config: DryConfiguration, exempt: Set<Int>) {
        self.config = config
        self.exempt = exempt
    }

    func update(_ token: Int) {
        guard !exempt.contains(token) else { return }
        history.append(token)
        let extra = history.count - config.historySize
        if extra > 0 { history.removeFirst(extra) }
        penalties = dryPenalties(history, config).filter { !exempt.contains($0.key) }
    }

    /// `logits [1, vocab]` minus the current penalties.
    func apply(_ logits: MLXArray) -> MLXArray {
        guard !penalties.isEmpty else { return logits }
        let ids = MLXArray(penalties.keys.map(Int32.init)).reshaped([1, -1])
        let values = MLXArray(penalties.keys.map { penalties[$0]! }).reshaped([1, -1])
        return putAlong(logits, ids, values: takeAlong(logits, ids, axis: -1) - values, axis: -1)
    }
}

/// Incremental text of the model's text stream: a run of PAD / EOP renders as one space, and a
/// multi-byte character is held back until all of its tokens have arrived.
final class StreamDetokenizer {
    let tokenizer: HaruTokenizer
    let silent: Set<Int>
    private var pending: [Int] = []
    private var inGap = false

    init(_ tokenizer: HaruTokenizer, silent: Set<Int>) {
        self.tokenizer = tokenizer
        self.silent = silent
    }

    func push(_ token: Int) -> String {
        if silent.contains(token) {
            if inGap { return "" }
            inGap = true
            return " "
        }
        inGap = false
        pending.append(token)
        let text = tokenizer.decode(pending)
        if text.hasSuffix("\u{FFFD}") { return "" }
        pending.removeAll()
        return text
    }
}
