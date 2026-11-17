// GatedDeltaRule.swift - the recurrence inside a Gated DeltaNet layer
// Copyright 2026 Vincent Gourbin
//
// Port of haru_mlx/models/gated_delta.py (higurashi-ai/Haru @ 7fe6b76, Apache-2.0; derived from
// mlx-lm, © Apple, MIT). Each value head keeps a fast-weight matrix S [Dv, Dk]:
//     S = g * S ; S = S + (beta * (v - S k)) kᵀ ; y = S q
// The Metal kernel is upstream's, copied as is: NOT mlx-swift-lm's, which adds a Kahan summation
// and gives different roundings (pitfall 3).

import Foundation
import MLX

enum GatedDeltaRule {
    static let source = """
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        // q, k: [B, T, Hk, Dk]
        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

        // v, y: [B, T, Hv, Dv]
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }

        // g, beta: [B, T, Hv]
        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {
          float kv_mem = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] * g_[hv_idx];
            kv_mem += state[i] * k_[s_idx];
          }
          kv_mem = simd_sum(kv_mem);

          auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

          float out = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] + k_[s_idx] * delta;
            out += state[i] * q_[s_idx];
          }
          out = simd_sum(out);
          if (thread_index_in_simdgroup == 0) {
            y[dv_idx] = static_cast<InT>(out);
          }
          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<StT>(state[i]);
        }
        """

    nonisolated(unsafe) static let kernel = MLXFast.metalKernel(
        name: "gated_delta_rule",
        inputNames: ["q", "k", "v", "g", "beta", "state_in", "T"],
        outputNames: ["y", "state_out"],
        source: source
    )

    /// Forces the array path (tests compare it with the kernel).
    nonisolated(unsafe) static var useKernel = true

    /// `q, k [B, T, Hk, Dk]`, `v [B, T, Hv, Dv]`, `g, beta [B, T, Hv]`, `state [B, Hv, Dv, Dk]` (fp32)
    /// -> `y [B, T, Hv, Dv]` in the dtype of `q`, and the new state.
    static func apply(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, _ g: MLXArray, _ beta: MLXArray,
                      _ state: MLXArray) -> (MLXArray, MLXArray) {
        useKernel && Device.defaultDevice().deviceType == .gpu
            ? metal(q, k, v, g, beta, state) : arrays(q, k, v, g, beta, state)
    }

    static func metal(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, _ g: MLXArray, _ beta: MLXArray,
                      _ state: MLXArray) -> (MLXArray, MLXArray) {
        let batch = k.dim(0), steps = k.dim(1), keyHeads = k.dim(2), keyDim = k.dim(3)
        let valueHeads = v.dim(2), valueDim = v.dim(3)
        let outputs = kernel(
            [q, k, v, g, beta, state, MLXArray(Int32(steps))],
            template: [("InT", q.dtype), ("StT", state.dtype), ("Dk", keyDim), ("Dv", valueDim),
                       ("Hk", keyHeads), ("Hv", valueHeads)],
            grid: (32, valueDim, batch * valueHeads),
            threadGroup: (32, 4, 1),
            outputShapes: [[batch, steps, valueHeads, valueDim], state.shape],
            outputDTypes: [q.dtype, state.dtype]
        )
        return (outputs[0], outputs[1])
    }

    static func arrays(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, _ g: MLXArray, _ beta: MLXArray,
                       _ state: MLXArray) -> (MLXArray, MLXArray) {
        var q = q, k = k, state = state
        let repeats = v.dim(2) / k.dim(2)
        if repeats > 1 {
            q = MLX.repeated(q, count: repeats, axis: 2)
            k = MLX.repeated(k, count: repeats, axis: 2)
        }
        var outputs: [MLXArray] = []
        for t in 0..<q.dim(1) {
            let key = expandedDimensions(k[0..., t], axis: -2)  // [B, H, 1, Dk]
            state = state * expandedDimensions(g[0..., t], axes: [-1, -2])
            let delta = (v[0..., t] - (state * key).sum(axis: -1)) * expandedDimensions(beta[0..., t], axis: -1)
            state = state + key * expandedDimensions(delta, axis: -1)
            outputs.append((state * expandedDimensions(q[0..., t], axis: -2)).sum(axis: -1))
        }
        return (stacked(outputs, axis: 1).asType(q.dtype), state)
    }
}
