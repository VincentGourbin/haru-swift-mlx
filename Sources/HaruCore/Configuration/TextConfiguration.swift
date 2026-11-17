// TextConfiguration.swift - the Qwen3.5 text backbone of a Haru checkpoint (`lm` in config.json)
// Copyright 2026 Vincent Gourbin
//
// Mirrors haru_mlx/lm_config.py (higurashi-ai/Haru @ 7fe6b76).

import Foundation

public struct TextConfiguration: Decodable, Sendable, Equatable {
    public enum LayerType: String, Codable, Sendable {
        case linearAttention = "linear_attention"
        case fullAttention = "full_attention"
    }

    public var vocabSize: Int
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var headDim: Int
    public var rmsNormEps: Float
    public var layerTypes: [LayerType]
    public var linearConvKernelDim: Int
    public var linearKeyHeadDim: Int
    public var linearValueHeadDim: Int
    public var linearKeyHeads: Int
    public var linearValueHeads: Int
    public var partialRotaryFactor: Float
    public var ropeTheta: Float
    public var eosTokenId: Int?

    // Derived sizes of the Gated DeltaNet layers (lm_config.py).
    public var keyDim: Int { linearKeyHeads * linearKeyHeadDim }
    public var valueDim: Int { linearValueHeads * linearValueHeadDim }
    /// Channels of the causal convolution: queries, keys and values.
    public var convDim: Int { 2 * keyDim + valueDim }
    /// Rotary dimensions of an attention head (64 of 256 for Haru-4B).
    public var rotaryDim: Int { Int(Float(headDim) * partialRotaryFactor) }

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size", hiddenSize = "hidden_size", intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers", numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads", headDim = "head_dim", rmsNormEps = "rms_norm_eps"
        case layerTypes = "layer_types", linearConvKernelDim = "linear_conv_kernel_dim"
        case linearKeyHeadDim = "linear_key_head_dim", linearValueHeadDim = "linear_value_head_dim"
        case linearKeyHeads = "linear_num_key_heads", linearValueHeads = "linear_num_value_heads"
        case partialRotaryFactor = "partial_rotary_factor", ropeParameters = "rope_parameters"
        case ropeTheta = "rope_theta", eosTokenId = "eos_token_id"
    }

    enum RopeKeys: String, CodingKey {
        case ropeTheta = "rope_theta", partialRotaryFactor = "partial_rotary_factor"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
        numAttentionHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
        numKeyValueHeads = try c.decode(Int.self, forKey: .numKeyValueHeads)
        headDim = try c.decode(Int.self, forKey: .headDim)
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        layerTypes = try c.decode([LayerType].self, forKey: .layerTypes)
        linearConvKernelDim = try c.decode(Int.self, forKey: .linearConvKernelDim)
        linearKeyHeadDim = try c.decode(Int.self, forKey: .linearKeyHeadDim)
        linearValueHeadDim = try c.decode(Int.self, forKey: .linearValueHeadDim)
        linearKeyHeads = try c.decode(Int.self, forKey: .linearKeyHeads)
        linearValueHeads = try c.decode(Int.self, forKey: .linearValueHeads)
        eosTokenId = try c.decodeIfPresent(Int.self, forKey: .eosTokenId)
        // rope_parameters wins over the top-level keys, as in lm_config.py.
        let rope = try? c.nestedContainer(keyedBy: RopeKeys.self, forKey: .ropeParameters)
        ropeTheta = try rope?.decodeIfPresent(Float.self, forKey: .ropeTheta)
            ?? c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000_000
        partialRotaryFactor = try rope?.decodeIfPresent(Float.self, forKey: .partialRotaryFactor)
            ?? c.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 0.25
    }
}
