// DepthwiseMixer.swift - `x + gamma * (depthwise_conv(concat(state, n)) + bias)` and the next state
// Copyright 2026 Vincent Gourbin
//
// Port of `depthwise_mixer` in haru_mlx/models/vibevoice_codec.py (higurashi-ai/Haru @ 7fe6b76,
// Apache-2.0): one Metal kernel on the GPU, accumulating in float32, with the same arithmetic in
// array operations as the fallback (CPU, and the reference the kernel is tested against).

import Foundation
import MLX

enum DepthwiseMixer {
    static let kernelSize = 7

    static let source = """
        uint c = thread_position_in_grid.x;
        uint r = thread_position_in_grid.y;
        if (c >= C) return;
        if (r < T) {
            float acc = float(bias[c]);
            for (int k = 0; k < K; ++k) {
                int i = int(r) + k - (K - 1);
                float v = i < 0 ? float(state[(i + K - 1) * C + c]) : float(n[i * C + c]);
                acc += float(w[c * K + k]) * v;
            }
            out[r * C + c] = static_cast<OutT>(float(x[r * C + c]) + float(gamma[c]) * acc);
        } else {
            int src = int(r);  // row of concat(state, n) that becomes row r - T of the next state
            new_state[(r - T) * C + c] = src < K - 1 ? state[src * C + c] : n[(src - (K - 1)) * C + c];
        }
        """

    nonisolated(unsafe) static let kernel = MLXFast.metalKernel(
        name: "causal_depthwise_mixer",
        inputNames: ["x", "n", "state", "w", "bias", "gamma"],
        outputNames: ["out", "new_state"],
        source: source
    )

    /// Forces the array path (tests compare it with the kernel).
    nonisolated(unsafe) static var useKernel = true

    static func apply(_ x: MLXArray, _ n: MLXArray, _ state: MLXArray, _ weight: MLXArray,
                      _ bias: MLXArray, _ gamma: MLXArray) -> (MLXArray, MLXArray) {
        useKernel && Device.defaultDevice().deviceType == .gpu
            ? metal(x, n, state, weight, bias, gamma)
            : arrays(x, n, state, weight, bias, gamma)
    }

    static func metal(_ x: MLXArray, _ n: MLXArray, _ state: MLXArray, _ weight: MLXArray,
                      _ bias: MLXArray, _ gamma: MLXArray) -> (MLXArray, MLXArray) {
        let steps = x.dim(0), channels = x.dim(1), k = weight.dim(1)
        let outputs = kernel(
            [x, n, state, weight, bias, gamma],
            template: [("OutT", x.dtype), ("T", steps), ("C", channels), ("K", k)],
            grid: (channels, steps + k - 1, 1),
            threadGroup: (min(channels, 256), 1, 1),
            outputShapes: [[steps, channels], [k - 1, channels]],
            outputDTypes: [x.dtype, state.dtype]
        )
        return (outputs[0], outputs[1])
    }

    static func arrays(_ x: MLXArray, _ n: MLXArray, _ state: MLXArray, _ weight: MLXArray,
                       _ bias: MLXArray, _ gamma: MLXArray) -> (MLXArray, MLXArray) {
        let steps = x.dim(0), k = weight.dim(1)
        let context = concatenated([state, n], axis: 0)
        let taps = context.asType(.float32)
        let w = weight.asType(.float32)
        var acc = bias.asType(.float32)
        for i in 0..<k {
            acc = acc + taps[i ..< (i + steps)] * w[0..., i]
        }
        let out = x.asType(.float32) + gamma.asType(.float32) * acc
        return (out.asType(x.dtype), context[steps...])
    }
}
