// HaruConfiguration.swift - config.json of a Haru checkpoint
// Copyright 2026 Vincent Gourbin
//
// Mirrors haru_mlx/config.py and lm_config.py (higurashi-ai/Haru @ 7fe6b76). Defaults are the
// published Haru-4B values; unknown keys are ignored, as `_pick` does upstream.

import Foundation

/// VibeVoice acoustic codec: 24 kHz waveform <-> 64-dim continuous latents at 7.5 Hz.
public struct CodecConfiguration: Codable, Sendable, Equatable {
    public var latentDim = 64
    public var baseChannels = 32
    /// Upsampling order (decoder); the encoder downsamples in reverse.
    public var ratios = [8, 5, 5, 4, 2, 2]
    public var encoderDepths = [3, 3, 3, 3, 3, 3, 8]
    public var decoderDepths = [8, 3, 3, 3, 3, 3, 3]
    public var normEps: Float = 1e-5
    public var latentStd: Float = 0.5

    /// Samples per latent: the product of the ratios (3200 for Haru-4B).
    public var frameSamples: Int { ratios.reduce(1, *) }

    enum CodingKeys: String, CodingKey {
        case latentDim = "latent_dim", baseChannels = "base_channels", ratios
        case encoderDepths = "encoder_depths", decoderDepths = "decoder_depths"
        case normEps = "norm_eps", latentStd = "latent_std"
    }

    public init() {}

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let d = CodecConfiguration()
        latentDim = try c.decodeIfPresent(Int.self, forKey: .latentDim) ?? d.latentDim
        baseChannels = try c.decodeIfPresent(Int.self, forKey: .baseChannels) ?? d.baseChannels
        ratios = try c.decodeIfPresent([Int].self, forKey: .ratios) ?? d.ratios
        encoderDepths = try c.decodeIfPresent([Int].self, forKey: .encoderDepths) ?? d.encoderDepths
        decoderDepths = try c.decodeIfPresent([Int].self, forKey: .decoderDepths) ?? d.decoderDepths
        normEps = try c.decodeIfPresent(Float.self, forKey: .normEps) ?? d.normEps
        latentStd = try c.decodeIfPresent(Float.self, forKey: .latentStd) ?? d.latentStd
    }
}

/// MeanFlow head: LM hidden state -> next audio latent, one sampling step.
public struct HeadConfiguration: Codable, Sendable, Equatable {
    public var hiddenSize = 2560
    public var latentDim = 64
    public var numLayers = 4
    public var ffnRatio: Float = 3.0
    public var normEps: Float = 1e-5
    /// Value fed to `dt_embed` (a constant of the trained model, not the step size).
    public var intervalEmbed: Float = 7.0
    public var numSteps = 1
    public var guidanceScale: Float = 1.5
    public var guidanceRange: [Float] = [1.0, 2.0]

    public var ffnSize: Int { Int(Float(hiddenSize) * ffnRatio) }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size", latentDim = "latent_dim", numLayers = "num_layers"
        case ffnRatio = "ffn_ratio", normEps = "norm_eps", intervalEmbed = "interval_embed"
        case numSteps = "num_steps", guidanceScale = "guidance_scale", guidanceRange = "guidance_range"
    }

    public init() {}

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let d = HeadConfiguration()
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? d.hiddenSize
        latentDim = try c.decodeIfPresent(Int.self, forKey: .latentDim) ?? d.latentDim
        numLayers = try c.decodeIfPresent(Int.self, forKey: .numLayers) ?? d.numLayers
        ffnRatio = try c.decodeIfPresent(Float.self, forKey: .ffnRatio) ?? d.ffnRatio
        normEps = try c.decodeIfPresent(Float.self, forKey: .normEps) ?? d.normEps
        intervalEmbed = try c.decodeIfPresent(Float.self, forKey: .intervalEmbed) ?? d.intervalEmbed
        numSteps = try c.decodeIfPresent(Int.self, forKey: .numSteps) ?? d.numSteps
        guidanceScale = try c.decodeIfPresent(Float.self, forKey: .guidanceScale) ?? d.guidanceScale
        guidanceRange = try c.decodeIfPresent([Float].self, forKey: .guidanceRange) ?? d.guidanceRange
    }
}

/// The whole checkpoint configuration.
public struct HaruConfiguration: Decodable, Sendable, Equatable {
    public var modelType: String
    public var sampleRate: Int
    public var frameSamples: Int
    public var latentDim: Int
    public var padTokenId: Int
    public var eopTokenId: Int
    public var lm: TextConfiguration
    public var codec: CodecConfiguration
    public var head: HeadConfiguration

    /// 7.5 Hz for Haru-4B: one language-model position per 133 ms.
    public var frameRate: Double { Double(sampleRate) / Double(frameSamples) }
    public var hiddenSize: Int { lm.hiddenSize }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type", sampleRate = "sample_rate", frameSamples = "frame_samples"
        case latentDim = "latent_dim", padTokenId = "pad_token_id", eopTokenId = "eop_token_id"
        case lm, codec, head
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "haru"
        sampleRate = try c.decodeIfPresent(Int.self, forKey: .sampleRate) ?? 24_000
        frameSamples = try c.decodeIfPresent(Int.self, forKey: .frameSamples) ?? 3200
        latentDim = try c.decodeIfPresent(Int.self, forKey: .latentDim) ?? 64
        padTokenId = try c.decodeIfPresent(Int.self, forKey: .padTokenId) ?? 248_076
        eopTokenId = try c.decodeIfPresent(Int.self, forKey: .eopTokenId) ?? 248_071
        lm = try c.decode(TextConfiguration.self, forKey: .lm)
        codec = try c.decodeIfPresent(CodecConfiguration.self, forKey: .codec) ?? CodecConfiguration()
        head = try c.decodeIfPresent(HeadConfiguration.self, forKey: .head) ?? HeadConfiguration()
    }

    public static func load(from data: Data) throws -> HaruConfiguration {
        let config: HaruConfiguration
        do {
            config = try JSONDecoder().decode(HaruConfiguration.self, from: data)
        } catch {
            throw HaruError.invalidConfiguration("\(error)")
        }
        try config.validate()
        return config
    }

    public static func load(contentsOf url: URL) throws -> HaruConfiguration {
        guard let data = FileManager.default.contents(atPath: url.path) else {
            throw HaruError.missingFile(url.lastPathComponent)
        }
        return try load(from: data)
    }

    /// The checks `HaruConfig.from_dict` makes, plus the layout facts the port relies on.
    public func validate() throws {
        guard modelType == "haru" else {
            throw HaruError.invalidConfiguration("not a Haru checkpoint (model_type=\(modelType))")
        }
        guard codec.frameSamples == frameSamples else {
            throw HaruError.invalidConfiguration("codec ratios give \(codec.frameSamples) samples, frame_samples is \(frameSamples)")
        }
        guard codec.latentDim == latentDim, head.latentDim == latentDim, head.hiddenSize == lm.hiddenSize else {
            throw HaruError.invalidConfiguration("latent or hidden sizes disagree between lm, head and codec")
        }
        guard lm.layerTypes.count == lm.numHiddenLayers else {
            throw HaruError.invalidConfiguration("\(lm.layerTypes.count) layer types for \(lm.numHiddenLayers) layers")
        }
        guard lm.linearValueHeads % lm.linearKeyHeads == 0, lm.linearKeyHeadDim % 32 == 0 else {
            throw HaruError.invalidConfiguration("Gated DeltaNet heads do not fit the Metal kernel (Hv % Hk, Dk % 32)")
        }
    }
}
