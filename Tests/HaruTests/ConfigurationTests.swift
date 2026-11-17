// ConfigurationTests.swift - config.json parsing and validation (weight-free)
// Copyright 2026 Vincent Gourbin

import Foundation
import Testing
@testable import HaruCore

@Suite struct ConfigurationTests {
    /// The `config.json` of higurashi-ai/Haru-4B, trimmed of keys the port ignores.
    static let published = """
    {"model_type": "haru", "sample_rate": 24000, "frame_samples": 3200, "frame_rate": 7.5,
     "latent_dim": 64, "pad_token_id": 248076, "eop_token_id": 248071,
     "lm": {"model_type": "qwen3_5_text", "vocab_size": 248320, "hidden_size": 2560,
            "intermediate_size": 9216, "num_hidden_layers": 32, "num_attention_heads": 16,
            "num_key_value_heads": 4, "head_dim": 256, "rms_norm_eps": 1e-06,
            "layer_types": [\(String(repeating: "\"linear_attention\", \"linear_attention\", \"linear_attention\", \"full_attention\", ", count: 7))"linear_attention", "linear_attention", "linear_attention", "full_attention"],
            "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_value_head_dim": 128,
            "linear_num_key_heads": 16, "linear_num_value_heads": 32,
            "rope_parameters": {"partial_rotary_factor": 0.25, "rope_theta": 10000000},
            "tie_word_embeddings": true, "eos_token_id": 248044},
     "codec": {"latent_dim": 64, "base_channels": 32, "ratios": [8, 5, 5, 4, 2, 2],
               "encoder_depths": [3, 3, 3, 3, 3, 3, 8], "decoder_depths": [8, 3, 3, 3, 3, 3, 3],
               "norm_eps": 1e-05, "latent_std": 0.5},
     "head": {"hidden_size": 2560, "latent_dim": 64, "num_layers": 4, "ffn_ratio": 3.0,
              "norm_eps": 1e-05, "interval_embed": 7.0, "num_steps": 1, "guidance_scale": 1.5,
              "guidance_range": [1.0, 2.0]}}
    """

    @Test func parsesThePublishedConfiguration() throws {
        let config = try HaruConfiguration.load(from: Data(Self.published.utf8))
        #expect(config.frameRate == 7.5)
        #expect(config.codec.frameSamples == 3200)
        #expect(config.lm.layerTypes.count == 32)
        #expect(config.lm.layerTypes.filter { $0 == .fullAttention }.count == 8)
        #expect(config.lm.convDim == 8192)
        #expect(config.lm.valueDim == 4096)
        #expect(config.lm.rotaryDim == 64)
        #expect(config.lm.eosTokenId == 248_044)
        #expect(config.head.ffnSize == 7680)
    }

    @Test func rejectsAnotherModelType() {
        let json = Self.published.replacingOccurrences(of: "\"model_type\": \"haru\"", with: "\"model_type\": \"moshi\"")
        #expect(throws: HaruError.self) { try HaruConfiguration.load(from: Data(json.utf8)) }
    }

    @Test func rejectsRatiosThatDoNotMakeAFrame() {
        let json = Self.published.replacingOccurrences(of: "\"frame_samples\": 3200", with: "\"frame_samples\": 1920")
        #expect(throws: HaruError.self) { try HaruConfiguration.load(from: Data(json.utf8)) }
    }

    @Test func inventorySplitsTheCheckpointByPart() throws {
        let header = SafetensorsHeader(entries: [
            "lm.embed_tokens.weight": .init(dtype: "BF16", shape: [10, 4]),
            "lm.layers.0.mlp.gate_proj.weight": .init(dtype: "BF16", shape: [8, 4]),
            "head.ada.weight": .init(dtype: "BF16", shape: [4, 4]),
            "codec.decoder.stem.weight": .init(dtype: "BF16", shape: [2, 3, 7]),
            "latent_scale": .init(dtype: "F32", shape: []),
        ])
        let inventory = WeightInventory(header: header)
        #expect(inventory.lines.map(\.part) == [.embedding, .backbone, .head, .codecDecoder, .other])
        #expect(inventory.totalParameters == 40 + 32 + 16 + 42 + 1)
        #expect(inventory.totalBytes == 2 * (40 + 32 + 16 + 42) + 4)
    }
}
