// VibeVoiceCodec.swift - 24 kHz audio <-> 64-dim latents at 7.5 Hz, both directions causal
// Copyright 2026 Vincent Gourbin
//
// Port of `Encoder`, `Decoder` and `VibeVoiceCodec` in haru_mlx/models/vibevoice_codec.py
// (higurashi-ai/Haru @ 7fe6b76, Apache-2.0; VibeVoice architecture © Microsoft, MIT).

import Foundation
import MLX

public final class VibeVoiceCodec {
    public let config: CodecConfiguration
    let dtype: DType
    /// `stem, stage0, down0, stage1, …, down5, stage6, head`.
    let encoder: [CodecLayer]
    /// `stem, stage0, up0, stage1, …, up5, stage6, head`.
    let decoder: [CodecLayer]
    /// The encoder is split at its last downsampling: the front runs on every `frontHop` samples while
    /// a frame's audio is still arriving, the back on whole frames.
    let frontLayers: Int
    public let frameHop: Int
    public let frontHop: Int

    init(_ store: WeightStore, config: CodecConfiguration) {
        self.config = config
        dtype = store.dtype
        let c = config.baseChannels, eps = config.normEps
        let strides = Array(config.ratios.reversed())
        var enc: [CodecLayer] = []
        for (i, depth) in config.encoderDepths.enumerated() {
            enc.append(i == 0
                ? CausalConv1d(store, "codec.encoder.stem", inChannels: 1, kernel: 7)
                : CausalConv1d(store, "codec.encoder.down.\(i - 1)", inChannels: c << (i - 1),
                               kernel: 2 * strides[i - 1], stride: strides[i - 1]))
            for b in 0..<depth { enc.append(CodecBlock(store, "codec.encoder.stages.\(i).\(b)", dim: c << i, eps: eps)) }
        }
        enc.append(CausalConv1d(store, "codec.encoder.head", inChannels: c << strides.count, kernel: 7))
        encoder = enc

        let n = config.ratios.count
        var dec: [CodecLayer] = []
        for (i, depth) in config.decoderDepths.enumerated() {
            dec.append(i == 0
                ? CausalConv1d(store, "codec.decoder.stem", inChannels: config.latentDim, kernel: 7)
                : CausalConvTranspose1d(store, "codec.decoder.up.\(i - 1)", outChannels: c << (n - i),
                                        stride: config.ratios[i - 1]))
            for b in 0..<depth { dec.append(CodecBlock(store, "codec.decoder.stages.\(i).\(b)", dim: c << (n - i), eps: eps)) }
        }
        dec.append(CausalConv1d(store, "codec.decoder.head", inChannels: c, kernel: 7))
        decoder = dec

        frontLayers = encoder.count - 2 - config.encoderDepths.last!
        frameHop = config.frameSamples
        frontHop = frameHop / strides.last!
    }

    /// Every parameter (to evaluate the codec once after loading).
    var arrays: [MLXArray] {
        (encoder + decoder).flatMap { layer -> [MLXArray] in
            switch layer {
            case let l as CausalConv1d: [l.weight, l.bias]
            case let l as CausalConvTranspose1d: [l.weight, l.bias]
            case let l as CodecBlock: [l.norm, l.filter, l.filterBias, l.gamma, l.ffnNorm, l.ffnGamma] + l.fc1.arrays + l.fc2.arrays
            default: []
            }
        }
    }

    // MARK: - States

    func initState() -> (encoder: [MLXArray], decoder: [MLXArray]) {
        (encoder.map { MLXArray.zeros($0.stateShape, dtype: dtype) },
         decoder.map { MLXArray.zeros($0.stateShape, dtype: dtype) })
    }

    // MARK: - Pieces run by the stream

    func runFront(_ audio: MLXArray, _ state: [MLXArray]) -> (MLXArray, [MLXArray]) {
        runCodecLayers(encoder[..<frontLayers], audio.asType(dtype).reshaped([-1, 1]), state[...])
    }

    func runBack(_ x: MLXArray, _ state: [MLXArray]) -> (MLXArray, [MLXArray]) {
        let (mean, next) = runCodecLayers(encoder[frontLayers...], x, state[...])
        return (mean.asType(.float32), next)
    }

    func runDecoder(_ latents: MLXArray, _ state: [MLXArray]) -> (MLXArray, [MLXArray]) {
        let (x, next) = runCodecLayers(decoder[...], latents.asType(dtype), state[...])
        return (x.reshaped([-1]).asType(.float32), next)
    }

    // MARK: - Whole clips

    /// Latents around `mean [B, T, D]`, one noise scale per batch row (`codec.sample`).
    public func sample(_ mean: MLXArray, key: MLXArray) -> MLXArray {
        let (scaleKey, noiseKey) = MLXRandom.split(key: key)
        let scale = MLXRandom.normal([mean.dim(0), 1, 1], key: scaleKey)
        let noise = MLXRandom.normal(mean.shape, key: noiseKey)
        return mean.asType(.float32) + scale * (config.latentStd / 0.8) * noise
    }

    /// Whole clip `[T]` -> latent means `[1, ceil(T / 3200), D]` in float32.
    public func encode(_ audio: MLXArray) -> MLXArray {
        let (mean, _) = runCodecLayers(encoder[...], audio.asType(dtype).reshaped([-1, 1]), initState().encoder[...])
        return expandedDimensions(mean.asType(.float32), axis: 0)
    }

    /// Whole sequence `[1, T, D]` -> audio `[1, T * 3200]` in float32.
    public func decode(_ latents: MLXArray) -> MLXArray {
        let x = latents.asType(dtype).reshaped([-1, latents.dim(-1)])
        let (audio, _) = runCodecLayers(decoder[...], x, initState().decoder[...])
        return audio.reshaped([1, -1]).asType(.float32)
    }
}
