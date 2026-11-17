// CodecLayers.swift - the layers of the VibeVoice codec, one stream, channel-last [T, C]
// Copyright 2026 Vincent Gourbin
//
// Port of haru_mlx/models/vibevoice_codec.py (higurashi-ai/Haru @ 7fe6b76, Apache-2.0; VibeVoice
// architecture © Microsoft, MIT). Every layer takes and returns its left context explicitly, so a
// whole clip (zero state) and one frame at a time give the same result.

import Foundation
import MLX
import MLXNN

/// A codec layer: `(x [T, C], state) -> (y, next state)`.
protocol CodecLayer: AnyObject {
    var stateShape: [Int] { get }
    func callAsFunction(_ x: MLXArray, _ state: MLXArray) -> (MLXArray, MLXArray)
}

/// Dense convolution whose left padding is the state; `weight [out, kernel * in]`, taps side by side.
/// A strided convolution has kernel = 2 * stride.
final class CausalConv1d: CodecLayer {
    let weight: MLXArray
    let bias: MLXArray
    let inChannels: Int, kernel: Int, stride: Int
    var context: Int { kernel - stride }
    var stateShape: [Int] { [context, inChannels] }

    init(_ store: WeightStore, _ prefix: String, inChannels: Int, kernel: Int, stride: Int = 1) {
        precondition(stride == 1 || kernel == 2 * stride, "a strided convolution needs kernel = 2 * stride")
        weight = store.take("\(prefix).weight")
        bias = store.take("\(prefix).bias")
        self.inChannels = inChannels
        self.kernel = kernel
        self.stride = stride
    }

    func callAsFunction(_ x: MLXArray, _ state: MLXArray) -> (MLXArray, MLXArray) {
        var x = x
        let tail = x.dim(0) % stride
        if tail != 0 {  // final partial hop of a whole-clip encode
            x = padded(x, widths: [IntOrPair((0, stride - tail)), IntOrPair((0, 0))])
        }
        let context = concatenated([state, x], axis: 0)
        let windows: MLXArray
        if stride == 1 {
            let steps = x.dim(0)
            windows = concatenated((0..<kernel).map { context[$0 ..< ($0 + steps)] }, axis: 1)
        } else {
            let hops = context.reshaped([-1, stride * inChannels])
            let n = hops.dim(0)
            windows = concatenated([hops[0 ..< (n - 1)], hops[1 ..< n]], axis: 1)
        }
        let total = context.dim(0)
        return (addMM(bias, windows, weight.T), context[(total - self.context) ..< total])
    }
}

/// Transposed convolution with kernel = 2 * stride; the state is the overlap tail.
/// `weight [out * kernel, in]` maps one input position to its `kernel` output samples per channel.
final class CausalConvTranspose1d: CodecLayer {
    let weight: MLXArray
    let bias: MLXArray
    let outChannels: Int, stride: Int
    var stateShape: [Int] { [stride, outChannels] }

    init(_ store: WeightStore, _ prefix: String, outChannels: Int, stride: Int) {
        weight = store.take("\(prefix).weight")
        bias = store.take("\(prefix).bias")
        self.outChannels = outChannels
        self.stride = stride
    }

    func callAsFunction(_ x: MLXArray, _ state: MLXArray) -> (MLXArray, MLXArray) {
        let y = matmul(x, weight.T).reshaped([-1, outChannels, 2, stride]).transposed(0, 2, 3, 1)  // [T, 2, stride, out]
        let first = y[0..., 0], second = y[0..., 1]
        let steps = second.dim(0)
        let overlap = concatenated([expandedDimensions(state, axis: 0), second[0 ..< (steps - 1)]], axis: 0)
        let out = (first + overlap).reshaped([-1, outChannels]) + bias
        return (out, second[steps - 1])
    }
}

/// Depthwise causal mixer and an MLP, each with a scaled residual.
final class CodecBlock: CodecLayer {
    let norm: MLXArray, filter: MLXArray, filterBias: MLXArray, gamma: MLXArray
    let ffnNorm: MLXArray, fc1: Dense, fc2: Dense, ffnGamma: MLXArray
    let dim: Int, eps: Float
    var stateShape: [Int] { [DepthwiseMixer.kernelSize - 1, dim] }

    init(_ store: WeightStore, _ prefix: String, dim: Int, eps: Float) {
        norm = store.take("\(prefix).norm.weight")
        filter = store.take("\(prefix).conv.weight")
        filterBias = store.take("\(prefix).conv.bias")
        gamma = store.take("\(prefix).gamma")
        ffnNorm = store.take("\(prefix).ffn_norm.weight")
        fc1 = Dense(store, "\(prefix).fc1", bias: true)
        fc2 = Dense(store, "\(prefix).fc2", bias: true)
        ffnGamma = store.take("\(prefix).ffn_gamma")
        self.dim = dim
        self.eps = eps
    }

    func callAsFunction(_ x: MLXArray, _ state: MLXArray) -> (MLXArray, MLXArray) {
        let n = MLXFast.rmsNorm(x, weight: norm, eps: eps)
        let (mixed, next) = DepthwiseMixer.apply(x, n, state, filter, filterBias, gamma)
        let h = fc2(gelu(fc1(MLXFast.rmsNorm(mixed, weight: ffnNorm, eps: eps))))
        return (mixed + ffnGamma * h, next)
    }
}

/// Runs layers in order, threading their states.
func runCodecLayers(_ layers: ArraySlice<CodecLayer>, _ x: MLXArray, _ state: ArraySlice<MLXArray>) -> (MLXArray, [MLXArray]) {
    var x = x
    var next: [MLXArray] = []
    next.reserveCapacity(layers.count)
    for (layer, s) in zip(layers, state) {
        let (y, n) = layer(x, s)
        x = y
        next.append(n)
    }
    return (x, next)
}
