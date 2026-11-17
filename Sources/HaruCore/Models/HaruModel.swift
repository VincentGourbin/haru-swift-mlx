// HaruModel.swift - Haru: three input streams summed into one position of the backbone
// Copyright 2026 Vincent Gourbin
//
// Port of haru_mlx/models/haru.py (higurashi-ai/Haru @ 7fe6b76, Apache-2.0). One position is one
// 133 ms frame: the model's text token, its audio latent and the user's audio latent are summed. The
// hidden state at frame t gives frame t + 1: the text token through the tied output embedding, the
// audio latent through the MeanFlow head. Every call must come from the thread that loaded the model.

import Foundation
import MLX

/// Audio latent -> LM embedding space.
final class AudioProjector {
    let fc1: Dense, fc2: Dense, norm: MLXArray
    init(_ store: WeightStore, _ prefix: String) {
        fc1 = Dense(store, "\(prefix).fc1", bias: true)
        fc2 = Dense(store, "\(prefix).fc2", bias: true)
        norm = store.take("\(prefix).norm.weight")
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(MLXFast.rmsNorm(fc1(x), weight: norm, eps: 1e-6))
    }
    var arrays: [MLXArray] { fc1.arrays + fc2.arrays + [norm] }
}

public final class HaruModel {
    public let config: HaruConfiguration
    public let dtype: DType
    let lm: Qwen35Backbone
    let modelAudioProj: AudioProjector
    let userAudioProj: AudioProjector
    let head: MeanFlowHead
    public let codec: VibeVoiceCodec
    let latentScale: MLXArray
    let latentBias: MLXArray
    public private(set) var tokenizer: HaruTokenizer?
    public private(set) var quantizedBits: Int?

    init(_ store: WeightStore, config: HaruConfiguration) {
        self.config = config
        dtype = store.dtype
        lm = Qwen35Backbone(store, config.lm)
        modelAudioProj = AudioProjector(store, "model_audio_proj")
        userAudioProj = AudioProjector(store, "user_audio_proj")
        head = MeanFlowHead(store, config.head)
        codec = VibeVoiceCodec(store, config: config.codec)
        latentScale = store.take("latent_scale")
        latentBias = store.take("latent_bias")
        precondition(store.remaining.isEmpty, "parameters not used by the model: \(store.remaining.prefix(5))")
    }

    /// Loads a checkpoint directory (`config.json`, `model.safetensors`, `tokenizer.json`). `dtype` nil
    /// keeps the checkpoint's bf16; `quantize` 8 stores the backbone and the embedding table in 8 bits.
    /// Evaluates every part on the calling thread (pitfall 14), one part at a time (pitfall 15).
    public static func load(directory: URL, dtype: DType? = nil, quantize: Int? = nil) async throws -> HaruModel {
        let config = try HaruConfiguration.load(contentsOf: directory.appendingPathComponent("config.json"))
        let store = try WeightStore.load(checkpoint: directory.appendingPathComponent("model.safetensors"),
                                         config: config, dtype: dtype)
        let model = HaruModel(store, config: config)
        model.tokenizer = try await HaruTokenizer.load(directory: directory)
        for layer in model.lm.layers { eval(layer.arrays) }
        eval(model.lm.embedTokens.arrays + [model.lm.norm.weight, model.latentScale, model.latentBias])
        eval(model.modelAudioProj.arrays + model.userAudioProj.arrays + model.head.arrays + model.codec.arrays)
        if let quantize { try model.quantize(bits: quantize) }
        return model
    }

    /// 8-bit backbone matrices and embedding table, as upstream (`HaruModel.quantize`): norm gains, the
    /// Gated DeltaNet convolution and gates, the audio projectors, the head and the codec keep their precision.
    public func quantize(bits: Int = 8, groupSize: Int = 64) throws {
        guard bits == 8 else { throw HaruError.invalidRequest("only 8-bit quantization is available") }
        for layer in lm.layers {
            for dense in layer.dense { dense.quantize(bits: bits, groupSize: groupSize) }
            eval(layer.arrays)
        }
        lm.embedTokens.quantize(bits: bits, groupSize: groupSize)
        eval(lm.embedTokens.arrays)
        quantizedBits = bits
    }

    // MARK: - Latents

    /// Codec latents -> the normalized space of the head and the projectors.
    func scaleLatents(_ raw: MLXArray) -> MLXArray { (raw.asType(.float32) + latentBias) * latentScale }
    func unscaleLatents(_ normalized: MLXArray) -> MLXArray { normalized.asType(.float32) / latentScale - latentBias }

    // MARK: - Backbone

    public func newCache() -> BackboneCache { lm.newCache(dtype: dtype) }

    /// Sums the three streams: `tokens [B, T]` and raw latents `[B, T, D]` -> `[B, T, H]`. A mask
    /// `[B, T]` zeroes an audio stream at positions that have none.
    func embed(tokens: MLXArray, modelLatents: MLXArray? = nil, userLatents: MLXArray? = nil,
               modelMask: MLXArray? = nil, userMask: MLXArray? = nil) -> MLXArray {
        var x = lm.embedTokens.rows(tokens)
        for (proj, latents, mask) in [(modelAudioProj, modelLatents, modelMask), (userAudioProj, userLatents, userMask)] {
            guard let latents else { continue }
            var y = proj(scaleLatents(latents).asType(x.dtype))
            if let mask { y = y * expandedDimensions(mask, axis: -1).asType(y.dtype) }
            x = x + y
        }
        return x
    }

    /// New positions `[B, T, H]` on top of `cache` -> hidden states `[B, T, H]`.
    func forward(_ embeddings: MLXArray, _ cache: BackboneCache) -> MLXArray { lm(embeddings, cache) }

    /// Text logits `[..., vocab]` in float32.
    func logits(_ hidden: MLXArray) -> MLXArray { lm.embedTokens(hidden).asType(.float32) }

    /// Logits of a few `tokens` entirely in float32: the PAD / EOP decision is finer than bf16 (pitfall 8).
    func logitsFP32(_ hidden: MLXArray, tokens: MLXArray) -> MLXArray {
        matmul(hidden.asType(.float32), lm.embedTokens.rows(tokens).asType(.float32).T)
    }
}
