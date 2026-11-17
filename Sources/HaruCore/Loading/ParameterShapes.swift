// ParameterShapes.swift - every parameter of the Haru model and its shape
// Copyright 2026 Vincent Gourbin
//
// Port of `parameter_shapes` in haru_mlx/weights.py (higurashi-ai/Haru @ 7fe6b76, Apache-2.0):
// the shapes the converted checkpoint must have, derived from the configuration alone.

import Foundation

extension WeightPlan {
    public static func parameterShapes(config: HaruConfiguration) -> [String: [Int]] {
        let lm = config.lm
        let hidden = lm.hiddenSize, latent = config.latentDim
        var shapes: [String: [Int]] = [
            "lm.embed_tokens.weight": [lm.vocabSize, hidden],
            "lm.norm.weight": [hidden],
            "latent_scale": [],
            "latent_bias": [],
        ]
        for (i, kind) in lm.layerTypes.enumerated() {
            let layer = "lm.layers.\(i)"
            shapes["\(layer).input_layernorm.weight"] = [hidden]
            shapes["\(layer).post_attention_layernorm.weight"] = [hidden]
            shapes["\(layer).mlp.gate_up.weight"] = [2 * lm.intermediateSize, hidden]
            shapes["\(layer).mlp.down_proj.weight"] = [hidden, lm.intermediateSize]
            switch kind {
            case .linearAttention:
                let block = "\(layer).linear_attn"
                let heads = lm.linearValueHeads
                shapes["\(block).in_proj.weight"] = [lm.convDim + lm.valueDim + 2 * heads, hidden]
                shapes["\(block).conv1d.weight"] = [lm.convDim, lm.linearConvKernelDim, 1]
                shapes["\(block).A_log"] = [heads]
                shapes["\(block).dt_bias"] = [heads]
                shapes["\(block).norm.weight"] = [lm.linearValueHeadDim]
                shapes["\(block).out_proj.weight"] = [hidden, lm.valueDim]
            case .fullAttention:
                let block = "\(layer).self_attn"
                let queryDim = lm.numAttentionHeads * lm.headDim
                let keyDim = lm.numKeyValueHeads * lm.headDim
                // The queries carry the output gate: [query, gate] per head.
                shapes["\(block).qkv.weight"] = [2 * queryDim + 2 * keyDim, hidden]
                shapes["\(block).q_norm.weight"] = [lm.headDim]
                shapes["\(block).k_norm.weight"] = [lm.headDim]
                shapes["\(block).o_proj.weight"] = [hidden, queryDim]
            }
        }
        for proj in ["model_audio_proj", "user_audio_proj"] {
            shapes["\(proj).fc1.weight"] = [hidden, latent]
            shapes["\(proj).fc1.bias"] = [hidden]
            shapes["\(proj).norm.weight"] = [hidden]
            shapes["\(proj).fc2.weight"] = [hidden, hidden]
            shapes["\(proj).fc2.bias"] = [hidden]
        }

        let head = config.head
        let width = head.hiddenSize, ffn = head.ffnSize
        shapes["head.latent_proj.weight"] = [width, head.latentDim]
        shapes["head.cond_proj.weight"] = [width, width]
        for embed in ["t_embed", "dt_embed", "h_embed", "omega_embed"] {
            shapes["head.\(embed).fc1.weight"] = [width, 256]
            shapes["head.\(embed).fc2.weight"] = [width, width]
        }
        shapes["head.ada.weight"] = [(3 * head.numLayers + 2) * width, width]
        for i in 0..<head.numLayers {
            shapes["head.layers.\(i).norm.weight"] = [width]
            shapes["head.layers.\(i).ffn.gate_proj.weight"] = [ffn, width]
            shapes["head.layers.\(i).ffn.up_proj.weight"] = [ffn, width]
            shapes["head.layers.\(i).ffn.down_proj.weight"] = [width, ffn]
        }
        shapes["head.final.proj.weight"] = [head.latentDim, width]

        let codec = config.codec
        let base = codec.baseChannels, stages = codec.ratios.count
        func conv(_ name: String, _ input: Int, _ output: Int, _ kernel: Int) {
            shapes["\(name).weight"] = [output, kernel * input]
            shapes["\(name).bias"] = [output]
        }
        func stage(_ name: String, _ channels: Int, _ depth: Int) {
            for b in 0..<depth {
                let block = "\(name).\(b)"
                shapes["\(block).norm.weight"] = [channels]
                shapes["\(block).conv.weight"] = [channels, 7]
                shapes["\(block).conv.bias"] = [channels]
                shapes["\(block).gamma"] = [channels]
                shapes["\(block).ffn_norm.weight"] = [channels]
                shapes["\(block).fc1.weight"] = [4 * channels, channels]
                shapes["\(block).fc1.bias"] = [4 * channels]
                shapes["\(block).fc2.weight"] = [channels, 4 * channels]
                shapes["\(block).fc2.bias"] = [channels]
                shapes["\(block).ffn_gamma"] = [channels]
            }
        }
        conv("codec.encoder.stem", 1, base, 7)
        for (i, stride) in codec.ratios.reversed().enumerated() {
            conv("codec.encoder.down.\(i)", base << i, base << (i + 1), 2 * stride)
        }
        for (i, depth) in codec.encoderDepths.enumerated() { stage("codec.encoder.stages.\(i)", base << i, depth) }
        conv("codec.encoder.head", base << stages, codec.latentDim, 7)

        conv("codec.decoder.stem", codec.latentDim, base << stages, 7)
        for (i, stride) in codec.ratios.enumerated() {
            let output = base << (stages - i - 1)
            shapes["codec.decoder.up.\(i).weight"] = [output * 2 * stride, base << (stages - i)]
            shapes["codec.decoder.up.\(i).bias"] = [output]
        }
        for (i, depth) in codec.decoderDepths.enumerated() {
            stage("codec.decoder.stages.\(i)", base << (stages - i), depth)
        }
        conv("codec.decoder.head", base, 1, 7)
        return shapes
    }
}
