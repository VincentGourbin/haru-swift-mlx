// ParitySupport.swift - real-weight parity against haru_mlx dumps (Scripts/reference/dump_parity.py)
// Copyright 2026 Vincent Gourbin
//
// Gated on HARU_MODELS_DIR (checkpoint) and HARU_PARITY_DIR (the dumps, under .local-runs/parity:
// real-weight activations are never committed).

import Foundation
import MLX
@testable import HaruCore

enum Parity {
    static var enabled: Bool {
        let env = ProcessInfo.processInfo.environment
        return env["HARU_MODELS_DIR"] != nil && env["HARU_PARITY_DIR"] != nil
    }

    static func reference(_ dtype: String) throws -> [String: MLXArray] {
        let dir = ProcessInfo.processInfo.environment["HARU_PARITY_DIR"]!
        return try MLX.loadArrays(url: URL(fileURLWithPath: dir).appendingPathComponent("haru-\(dtype).safetensors"))
    }

    static var checkpoint: URL { HaruWeights.directory() }

    static func config() throws -> HaruConfiguration {
        try HaruConfiguration.load(contentsOf: checkpoint.appendingPathComponent("config.json"))
    }

    static func store(_ dtype: DType) throws -> WeightStore {
        try WeightStore.load(checkpoint: checkpoint.appendingPathComponent("model.safetensors"), config: config(), dtype: dtype)
    }

    /// ‖a − b‖ / ‖b‖ in float64 on the CPU side.
    static func relativeError(_ a: MLXArray, _ b: MLXArray) -> Double {
        precondition(a.shape == b.shape, "shapes \(a.shape) vs \(b.shape)")
        let x = a.asType(.float32).asArray(Float.self), y = b.asType(.float32).asArray(Float.self)
        var num = 0.0, den = 0.0
        for (p, q) in zip(x, y) {
            num += Double(p - q) * Double(p - q)
            den += Double(q) * Double(q)
        }
        return (num / max(den, 1e-30)).squareRoot()
    }
}
