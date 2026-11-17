// Qwen35Backbone.swift - 32 layers (3 Gated DeltaNet then 1 attention, ×8), one position per frame
// Copyright 2026 Vincent Gourbin
//
// Port of `DecoderLayer` and `Qwen3_5TextModel` in haru_mlx/models/qwen3_5.py (higurashi-ai/Haru @
// 7fe6b76, Apache-2.0).

import Foundation
import MLX

/// The per-layer state of one conversation.
public final class BackboneCache {
    var delta: [Int: DeltaState] = [:]
    var attention: [Int: KVCache] = [:]

    /// Positions run so far.
    public var offset: Int { attention.values.first?.offset ?? 0 }

    var arrays: [MLXArray] {
        delta.values.flatMap { [$0.conv, $0.memory] } + attention.values.flatMap(\.arrays)
    }

    /// An independent copy: the arrays are immutable values, so copying the references is enough
    /// as long as the KV buffers are not written in place by both (`KVCache` copies on write below).
    func copy() -> BackboneCache {
        let c = BackboneCache()
        c.delta = delta
        for (i, kv) in attention {
            let n = KVCache()
            n.keys = kv.keys.map { $0[0..., 0..., ..<kv.offset, 0...] }
            n.values = kv.values.map { $0[0..., 0..., ..<kv.offset, 0...] }
            n.offset = kv.offset
            c.attention[i] = n
        }
        return c
    }
}

final class DecoderLayer {
    let inputNorm: GainRMSNorm, postNorm: GainRMSNorm
    let linear: GatedDeltaNet?
    let attention: GatedAttention?
    let mlp: SwiGLU

    init(_ store: WeightStore, index: Int, kind: TextConfiguration.LayerType, _ c: TextConfiguration) {
        let p = "lm.layers.\(index)"
        inputNorm = GainRMSNorm(weight: store.take("\(p).input_layernorm.weight"), eps: c.rmsNormEps)
        postNorm = GainRMSNorm(weight: store.take("\(p).post_attention_layernorm.weight"), eps: c.rmsNormEps)
        linear = kind == .linearAttention ? GatedDeltaNet(store, "\(p).linear_attn", c) : nil
        attention = kind == .fullAttention ? GatedAttention(store, "\(p).self_attn", c) : nil
        mlp = SwiGLU(store, "\(p).mlp")
    }

    var dense: [Dense] {
        [mlp.gateUp, mlp.down] + (linear.map { [$0.inProj, $0.outProj] } ?? [])
            + (attention.map { [$0.qkv, $0.oProj] } ?? [])
    }

    var arrays: [MLXArray] {
        var all = [inputNorm.weight, postNorm.weight] + dense.flatMap(\.arrays)
        if let l = linear { all += [l.convWeight, l.aLog, l.dtBias, l.normWeight] }
        if let a = attention { all += [a.qNorm.weight, a.kNorm.weight] }
        return all
    }

    func callAsFunction(_ x: MLXArray, index: Int, _ cache: BackboneCache) -> MLXArray {
        let r: MLXArray
        if let linear {
            let (y, state) = linear(inputNorm(x), cache.delta[index]!)
            cache.delta[index] = state
            r = y
        } else {
            r = attention!(inputNorm(x), cache.attention[index]!)
        }
        let h = x + r
        return h + mlp(postNorm(h))
    }
}

public final class Qwen35Backbone {
    let config: TextConfiguration
    let embedTokens: Dense
    let layers: [DecoderLayer]
    let norm: GainRMSNorm

    init(_ store: WeightStore, _ config: TextConfiguration) {
        self.config = config
        embedTokens = Dense(weight: store.take("lm.embed_tokens.weight"))
        layers = config.layerTypes.enumerated().map { DecoderLayer(store, index: $0.offset, kind: $0.element, config) }
        norm = GainRMSNorm(weight: store.take("lm.norm.weight"), eps: config.rmsNormEps)
    }

    func newCache(batch: Int = 1, dtype: DType) -> BackboneCache {
        let cache = BackboneCache()
        for (i, layer) in layers.enumerated() {
            if let l = layer.linear { cache.delta[i] = l.initState(batch: batch, dtype: dtype) } else { cache.attention[i] = KVCache() }
        }
        return cache
    }

    /// New positions `[B, T, H]` on top of `cache` -> hidden states `[B, T, H]` (after the final norm).
    func callAsFunction(_ embeddings: MLXArray, _ cache: BackboneCache) -> MLXArray {
        var h = embeddings
        for (i, layer) in layers.enumerated() { h = layer(h, index: i, cache) }
        return norm(h)
    }
}
