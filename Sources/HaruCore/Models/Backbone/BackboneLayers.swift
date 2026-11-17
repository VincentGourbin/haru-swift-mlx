// BackboneLayers.swift - norms, MLP, Gated DeltaNet and attention of the Qwen3.5 backbone
// Copyright 2026 Vincent Gourbin
//
// Port of haru_mlx/models/qwen3_5.py (higurashi-ai/Haru @ 7fe6b76, Apache-2.0; derived from mlx-lm,
// © Apple, MIT), with its fused projections: GDN `in_proj` = qkv | z | b | a, attention `qkv` with
// [query, gate] per head, MLP `gate_up`.

import Foundation
import MLX
import MLXNN

/// `nn.softplus` as upstream: compiled (shapeless), so it fuses into one kernel like Python's
/// `@partial(mx.compile, shapeless=True)`; MLXNN's is not compiled and rounds differently.
nonisolated(unsafe) let compiledSoftplus: (MLXArray) -> MLXArray = compile(shapeless: true) { x in logAddExp(x, 0) }

/// RMS norm in float32 with a float32 gain stored as `1 + w` (pitfall 1).
struct GainRMSNorm {
    let weight: MLXArray
    let eps: Float
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x.asType(.float32), weight: weight, eps: eps).asType(x.dtype)
    }
}

/// SwiGLU feed-forward; `gate_up` stacks the gate and up projections.
final class SwiGLU {
    let gateUp: Dense, down: Dense
    init(_ store: WeightStore, _ prefix: String) {
        gateUp = Dense(store, "\(prefix).gate_up")
        down = Dense(store, "\(prefix).down_proj")
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let parts = split(gateUp(x), parts: 2, axis: -1)
        return down(silu(parts[0]) * parts[1])
    }
}

/// State of one Gated DeltaNet layer.
struct DeltaState {
    var conv: MLXArray    // [B, kernel - 1, conv_dim]
    var memory: MLXArray  // [B, Hv, Dv, Dk] float32
}

final class GatedDeltaNet {
    let keyHeads: Int, valueHeads: Int, keyHeadDim: Int, valueHeadDim: Int
    let keyDim: Int, valueDim: Int, convDim: Int, kernel: Int
    let inProj: Dense, outProj: Dense
    let convWeight: MLXArray  // [conv_dim, kernel, 1]
    let aLog: MLXArray, dtBias: MLXArray, normWeight: MLXArray
    let eps: Float

    init(_ store: WeightStore, _ prefix: String, _ c: TextConfiguration) {
        keyHeads = c.linearKeyHeads; valueHeads = c.linearValueHeads
        keyHeadDim = c.linearKeyHeadDim; valueHeadDim = c.linearValueHeadDim
        keyDim = c.keyDim; valueDim = c.valueDim; convDim = c.convDim; kernel = c.linearConvKernelDim
        inProj = Dense(store, "\(prefix).in_proj")
        outProj = Dense(store, "\(prefix).out_proj")
        convWeight = store.take("\(prefix).conv1d.weight")
        aLog = store.take("\(prefix).A_log")
        dtBias = store.take("\(prefix).dt_bias")
        normWeight = store.take("\(prefix).norm.weight")
        eps = c.rmsNormEps
    }

    func initState(batch: Int, dtype: DType) -> DeltaState {
        DeltaState(conv: MLXArray.zeros([batch, kernel - 1, convDim], dtype: dtype),
                   memory: MLXArray.zeros([batch, valueHeads, valueHeadDim, keyHeadDim], dtype: .float32))
    }

    func callAsFunction(_ x: MLXArray, _ state: DeltaState) -> (MLXArray, DeltaState) {
        let batch = x.dim(0), steps = x.dim(1)
        let zStart = convDim, bStart = zStart + valueDim
        let parts = split(inProj(x), indices: [zStart, bStart, bStart + valueHeads], axis: -1)
        let (qkv, z, b, a) = (parts[0], parts[1], parts[2], parts[3])

        let convInput = concatenated([state.conv, qkv], axis: 1)
        let total = convInput.dim(1)
        let nextConv = convInput[0..., (total - (kernel - 1))...]
        let mixed = silu(conv1d(convInput, convWeight, groups: convDim))
        let qkvParts = split(mixed, indices: [keyDim, 2 * keyDim], axis: -1)
        var q = qkvParts[0].reshaped([batch, steps, keyHeads, keyHeadDim])
        var k = qkvParts[1].reshaped([batch, steps, keyHeads, keyHeadDim])
        let v = qkvParts[2].reshaped([batch, steps, valueHeads, valueHeadDim])

        // Unit-length keys and queries, the queries scaled by 1 / sqrt(Dk) (pitfall 5).
        let scale = Float(pow(Double(keyHeadDim), -0.5))
        let normEps = 1e-6 * scale * scale
        q = (scale * scale) * plainRMSNorm(q, eps: normEps)
        k = scale * plainRMSNorm(k, eps: normEps)

        let beta = sigmoid(b)
        // float32: the decay compounds over the whole conversation (pitfall 4).
        let decay = exp(-exp(aLog.asType(.float32)) * compiledSoftplus(a.asType(.float32) + dtBias))
        let (y, memory) = GatedDeltaRule.apply(q, k, v, decay, beta, state.memory)

        // GatedNorm: rms_norm(y, w) then silu(z) * normed in float32.
        let normed = MLXFast.rmsNorm(y, weight: normWeight, eps: eps).asType(.float32)
        let gated = (silu(z.reshaped([batch, steps, valueHeads, valueHeadDim]).asType(.float32)) * normed).asType(y.dtype)
        return (outProj(gated.reshaped([batch, steps, -1])), DeltaState(conv: nextConv, memory: memory))
    }
}

/// Keys and values of one attention layer, grown in blocks of 256 positions.
final class KVCache {
    static let step = 256
    var keys: MLXArray?
    var values: MLXArray?
    var offset = 0

    func updateAndFetch(_ k: MLXArray, _ v: MLXArray) -> (MLXArray, MLXArray) {
        let start = offset, end = offset + k.dim(2)
        let capacity = keys?.dim(2) ?? 0
        if end > capacity {
            let extra = (end - capacity + Self.step - 1) / Self.step * Self.step
            keys = Self.extend(keys, like: k, extra: extra)
            values = Self.extend(values, like: v, extra: extra)
        }
        keys![0..., 0..., start ..< end, 0...] = k
        values![0..., 0..., start ..< end, 0...] = v
        offset = end
        return (keys![0..., 0..., ..<end, 0...], values![0..., 0..., ..<end, 0...])
    }

    static func extend(_ buffer: MLXArray?, like: MLXArray, extra: Int) -> MLXArray {
        let block = MLXArray.zeros([like.dim(0), like.dim(1), extra, like.dim(3)], dtype: like.dtype)
        return buffer.map { concatenated([$0, block], axis: 2) } ?? block
    }

    var arrays: [MLXArray] { [keys, values].compactMap { $0 } }
}

/// Grouped-query attention with partial rotary positions and an output gate.
final class GatedAttention {
    let heads: Int, kvHeads: Int, headDim: Int, rotaryDim: Int, ropeBase: Float, scale: Float
    let qkv: Dense, oProj: Dense
    let qNorm: GainRMSNorm, kNorm: GainRMSNorm

    init(_ store: WeightStore, _ prefix: String, _ c: TextConfiguration) {
        heads = c.numAttentionHeads; kvHeads = c.numKeyValueHeads; headDim = c.headDim
        rotaryDim = c.rotaryDim; ropeBase = c.ropeTheta
        scale = Float(pow(Double(headDim), -0.5))
        qkv = Dense(store, "\(prefix).qkv")
        oProj = Dense(store, "\(prefix).o_proj")
        qNorm = GainRMSNorm(weight: store.take("\(prefix).q_norm.weight"), eps: c.rmsNormEps)
        kNorm = GainRMSNorm(weight: store.take("\(prefix).k_norm.weight"), eps: c.rmsNormEps)
    }

    func callAsFunction(_ x: MLXArray, _ cache: KVCache) -> MLXArray {
        let batch = x.dim(0), steps = x.dim(1)
        let qEnd = 2 * heads * headDim
        let parts = split(qkv(x), indices: [qEnd, qEnd + kvHeads * headDim], axis: -1)
        let qg = split(parts[0].reshaped([batch, steps, heads, -1]), parts: 2, axis: -1)
        var q = qNorm(qg[0]).transposed(0, 2, 1, 3)
        var k = kNorm(parts[1].reshaped([batch, steps, kvHeads, -1])).transposed(0, 2, 1, 3)
        var v = parts[2].reshaped([batch, steps, kvHeads, -1]).transposed(0, 2, 1, 3)
        let gate = qg[1].reshaped([batch, steps, -1])

        // Non-interleaved RoPE on the first 64 dimensions, after the norms (pitfall 6).
        q = MLXFast.RoPE(q, dimensions: rotaryDim, traditional: false, base: ropeBase, scale: 1, offset: cache.offset)
        k = MLXFast.RoPE(k, dimensions: rotaryDim, traditional: false, base: ropeBase, scale: 1, offset: cache.offset)
        (k, v) = cache.updateAndFetch(k, v)
        let out = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale,
                                                    mask: steps > 1 ? .causal : .none)
        let attended = out.transposed(0, 2, 1, 3).reshaped([batch, steps, -1])
        return oProj(attended * sigmoid(gate))
    }
}
