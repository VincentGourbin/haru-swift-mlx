// Dense.swift - a linear layer `x Wᵀ + b`, optionally with 8-bit affine-quantized weights
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX

public final class Dense {
    public private(set) var weight: MLXArray
    public let bias: MLXArray?
    private var scales: MLXArray?
    private var biases: MLXArray?
    private var groupSize = 64
    private var bits = 8

    public init(weight: MLXArray, bias: MLXArray? = nil) {
        self.weight = weight
        self.bias = bias
    }

    public convenience init(_ store: WeightStore, _ prefix: String, bias: Bool = false) {
        self.init(weight: store.take("\(prefix).weight"), bias: bias ? store.take("\(prefix).bias") : nil)
    }

    public var isQuantized: Bool { scales != nil }
    public var outputDimensions: Int { isQuantized ? scales!.dim(0) : weight.dim(0) }

    /// Replaces the weight by its affine quantization (as `nn.quantize`, group 64).
    public func quantize(bits: Int = 8, groupSize: Int = 64) {
        let (q, s, b) = MLX.quantized(weight, groupSize: groupSize, bits: bits)
        weight = q
        scales = s
        biases = b
        self.bits = bits
        self.groupSize = groupSize
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y: MLXArray
        if let scales, let biases {
            y = quantizedMatmul(x, weight, scales: scales, biases: biases, transpose: true,
                                groupSize: groupSize, bits: bits)
        } else if let bias {
            return addMM(bias, x, weight.T)
        } else {
            y = matmul(x, weight.T)
        }
        if let bias { y = y + bias }
        return y
    }

    /// Rows `ids` of the weight as dense values (dequantized when needed): embedding lookup.
    public func rows(_ ids: MLXArray) -> MLXArray {
        guard let scales, let biases else { return weight[ids] }
        return dequantized(weight[ids], scales: scales[ids], biases: biases[ids], groupSize: groupSize, bits: bits)
    }

    public var arrays: [MLXArray] { [weight] + [bias, scales, biases].compactMap { $0 } }
}

/// `rms_norm(x, None, eps)`: MLX Swift wants a weight, and a gain of ones changes nothing.
func plainRMSNorm(_ x: MLXArray, eps: Float) -> MLXArray {
    MLXFast.rmsNorm(x, weight: MLXArray.ones([x.dim(-1)], dtype: x.dtype), eps: eps)
}
