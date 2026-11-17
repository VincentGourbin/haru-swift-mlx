// WeightStore.swift - the checkpoint converted to model parameters, taken by name
// Copyright 2026 Vincent Gourbin
//
// Applies the conversion plan of `WeightPlan` (port of haru_mlx/weights.py) and `loaders._convert`:
// fusions by concatenation on axis 0, fp32 `1 + w` gains, fp32 scalars, channel-last and matrix
// layouts of the convolutions, then a cast to the working dtype. Arrays stay lazy until the model
// evaluates each part (pitfall 15: never a global eval of every parameter at once).

import Foundation
import MLX

public final class WeightStore {
    private var parameters: [String: MLXArray]
    public let dtype: DType

    /// `parameters` already converted (tests, synthetic checkpoints).
    public init(parameters: [String: MLXArray], dtype: DType) {
        self.parameters = parameters
        self.dtype = dtype
    }

    /// Loads `model.safetensors` and converts every tensor; `dtype` nil keeps the checkpoint's (bf16).
    public static func load(checkpoint url: URL, config: HaruConfiguration, dtype: DType? = nil) throws -> WeightStore {
        let tensors = try MLX.loadArrays(url: url)
        let plan = try WeightPlan.conversionPlan(shapes: tensors.mapValues(\.shape))
        try WeightPlan.check(plan, expected: WeightPlan.parameterShapes(config: config))
        let working = dtype ?? tensors["lm.embed_tokens.weight"]?.dtype ?? .bfloat16
        var converted: [String: MLXArray] = [:]
        for (name, conversion) in plan {
            converted[name] = convert(conversion, tensors, dtype: working)
        }
        return WeightStore(parameters: converted, dtype: working)
    }

    static func convert(_ conversion: WeightConversion, _ tensors: [String: MLXArray], dtype: DType) -> MLXArray {
        let parts = conversion.sources.map { tensors[$0]! }
        var value = parts.count == 1 ? parts[0] : concatenated(parts, axis: 0)
        switch conversion.layout {
        case .gamma: return value.asType(.float32) + 1
        case .scalar: return value.asType(.float32).reshaped([])
        case .channelsLast: value = value.transposed(0, 2, 1)
        case .convMatrix: value = value.transposed(0, 2, 1).reshaped(conversion.shape)
        case .convTransposeMatrix: value = value.transposed(1, 2, 0).reshaped(conversion.shape)
        case .copy: break
        }
        return value.asType(dtype)
    }

    /// Removes and returns one parameter; a missing name is a programming error of the port.
    public func take(_ name: String) -> MLXArray {
        guard let value = parameters.removeValue(forKey: name) else {
            fatalError("WeightStore: no parameter \(name)")
        }
        return value
    }

    public func takeIfPresent(_ name: String) -> MLXArray? {
        parameters.removeValue(forKey: name)
    }

    /// Parameters not taken by any module (empty once the model is built).
    public var remaining: [String] { parameters.keys.sorted() }
}
