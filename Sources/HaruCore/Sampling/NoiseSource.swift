// NoiseSource.swift - the random stream of one conversation
// Copyright 2026 Vincent Gourbin
//
// haru_mlx/session.py keeps one MLX key per session and splits it before every draw
// (`self._key, key = mx.random.split(self._key)`); it never touches the global random state.
// mlx-swift runs the same C++ generator (threefry), so a seed gives the same draws in Swift as in
// Python on the same machine: checked bit for bit by `NoiseSourceTests` (card T-0.5).
//
// A NoiseSource holds MLX arrays: create and use it on the thread that owns the model.

import Foundation
import MLX

public final class NoiseSource {
    /// The key the next draw will be split from.
    public private(set) var state: MLXArray
    public let seed: UInt64

    public init(seed: UInt64) {
        self.seed = seed
        state = MLXRandom.key(seed)
    }

    /// A fresh seed for a conversation without one: 32 bits, as `os.urandom(4)` upstream.
    public static func randomSeed() -> UInt64 {
        UInt64(UInt32.random(in: .min ... .max))
    }

    /// `_next_key`: advance the state, return the key of one draw.
    public func nextKey() -> MLXArray {
        let (next, draw) = MLXRandom.split(key: state)
        state = next
        return draw
    }

    /// Standard normal noise drawn in float32, then cast (the head noise is cast to the model dtype).
    public func normal(_ shape: [Int], dtype: DType = .float32) -> MLXArray {
        let noise = MLXRandom.normal(shape, key: nextKey())
        return dtype == .float32 ? noise : noise.asType(dtype)
    }
}
