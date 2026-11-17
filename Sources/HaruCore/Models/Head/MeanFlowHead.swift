// MeanFlowHead.swift - one audio latent from one LM hidden state, in one sampling step
// Copyright 2026 Vincent Gourbin
//
// Port of haru_mlx/models/meanflow_head.py (higurashi-ai/Haru @ 7fe6b76, Apache-2.0). The head
// predicts the mean velocity u(z, t, h | c, w); sampling is z += h * u. The guidance w is an input of
// the network, so one pass of a small MLP gives a guided latent. The time, interval, step and guidance
// embeddings do not depend on the hidden state and are computed once per (steps, guidance, dtype).

import Foundation
import MLX
import MLXNN

final class MeanFlowHead {
    static let frequencyDim = 256

    struct ScalarEmbed {
        let fc1: Dense, fc2: Dense
        /// A scalar -> `[1, hidden]`; the scalar is rounded to `dtype` first (pitfall 9).
        func callAsFunction(_ value: Float, dtype: DType) -> MLXArray {
            let half = MeanFlowHead.frequencyDim / 2
            let freqs = exp(-Float(log(10_000.0)) * MLXArray(0 ..< half).asType(.float32) / Float(half))
            let args = MLXArray([value]).asType(dtype).asType(.float32).reshaped([1, 1]) * freqs
            let features = concatenated([cos(args), sin(args)], axis: -1).asType(dtype)
            return fc2(silu(fc1(features)))
        }
    }

    struct Layer {
        let norm: MLXArray
        let gate: Dense, up: Dense, down: Dense
    }

    let config: HeadConfiguration
    let latentProj: Dense, condProj: Dense, ada: Dense, finalProj: Dense
    let tEmbed: ScalarEmbed, dtEmbed: ScalarEmbed, hEmbed: ScalarEmbed, omegaEmbed: ScalarEmbed
    let layers: [Layer]
    private var conditioning: [String: (times: [MLXArray], extra: [MLXArray])] = [:]

    init(_ store: WeightStore, _ config: HeadConfiguration) {
        self.config = config
        latentProj = Dense(store, "head.latent_proj")
        condProj = Dense(store, "head.cond_proj")
        ada = Dense(store, "head.ada")
        finalProj = Dense(store, "head.final.proj")
        func embed(_ name: String) -> ScalarEmbed {
            ScalarEmbed(fc1: Dense(store, "head.\(name).fc1"), fc2: Dense(store, "head.\(name).fc2"))
        }
        tEmbed = embed("t_embed"); dtEmbed = embed("dt_embed"); hEmbed = embed("h_embed"); omegaEmbed = embed("omega_embed")
        layers = (0..<config.numLayers).map { i in
            Layer(norm: store.take("head.layers.\(i).norm.weight"),
                  gate: Dense(store, "head.layers.\(i).ffn.gate_proj"),
                  up: Dense(store, "head.layers.\(i).ffn.up_proj"),
                  down: Dense(store, "head.layers.\(i).ffn.down_proj"))
        }
    }

    var arrays: [MLXArray] {
        ([latentProj, condProj, ada, finalProj] + [tEmbed, dtEmbed, hEmbed, omegaEmbed].flatMap { [$0.fc1, $0.fc2] }
            + layers.flatMap { [$0.gate, $0.up, $0.down] }).flatMap(\.arrays) + layers.map(\.norm)
    }

    private func rmsNorm32(_ x: MLXArray) -> MLXArray {
        plainRMSNorm(x.asType(.float32), eps: config.normEps).asType(x.dtype)
    }

    /// `z [B, D]` and the summed conditioning `c [B, H]` -> mean velocity `[B, D]`.
    func velocity(_ z: MLXArray, _ c: MLXArray) -> MLXArray {
        let hidden = config.hiddenSize
        let modulations = split(ada(silu(c)), indices: (1 ... layers.count).map { 3 * hidden * $0 }, axis: -1)
        var x = latentProj(z)
        for (layer, modulation) in zip(layers, modulations) {
            let m = split(modulation, parts: 3, axis: -1)  // shift, scale, gate
            let h = rmsNorm32(x) * layer.norm * (1 + m[1]) + m[0]
            x = x + m[2] * layer.down(silu(layer.gate(h)) * layer.up(h))
        }
        let f = split(modulations.last!, parts: 2, axis: -1)
        return finalProj(rmsNorm32(x) * (1 + f[1]) + f[0])
    }

    /// LM hidden states `[B, H]` and noise `[B, D]` -> normalized latents `[B, D]`.
    func sample(_ hidden: MLXArray, noise: MLXArray, guidance: Float? = nil, steps: Int? = nil) -> MLXArray {
        let steps = steps ?? config.numSteps
        let guidance = guidance ?? config.guidanceScale
        let dtype = noise.dtype
        let key = "\(steps)/\(guidance)/\(dtype)"
        if conditioning[key] == nil {
            let h = 1.0 / Float(steps)
            let times = (0..<steps).map { tEmbed(Float($0) * h, dtype: dtype) }
            let extra = [dtEmbed(config.intervalEmbed, dtype: dtype), hEmbed(h, dtype: dtype),
                         omegaEmbed(1 - 1 / guidance, dtype: dtype)]
            eval(times + extra)
            conditioning[key] = (times, extra)
        }
        let (times, extra) = conditioning[key]!
        let h = 1.0 / Float(steps)
        let cond = condProj(hidden.asType(dtype))
        var z = noise
        for time in times {
            // Each sum rounds in the working dtype: the order is part of the result (pitfall 9).
            var c = cond + time
            c = c + extra[0]
            c = c + extra[1]
            c = c + extra[2]
            z = z + h * velocity(z, c)
        }
        return z
    }
}
