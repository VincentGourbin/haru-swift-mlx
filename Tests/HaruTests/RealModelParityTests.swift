// RealModelParityTests.swift - backbone, logits and MeanFlow head against haru_mlx on the real weights
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import Testing
@testable import HaruCore

@Suite(.serialized, .enabled(if: Parity.enabled))
struct RealModelParityTests {
    func run(_ dtype: DType, _ name: String) throws -> [String: Double] {
        let ref = try Parity.reference(name)
        let model = HaruModel(try Parity.store(dtype), config: try Parity.config())
        let text = ref["in.text"]!, modelLatents = ref["in.model_latents"]!, userLatents = ref["in.user_latents"]!
        let modelMask = ref["in.model_mask"]!, userMask = ref["in.user_mask"]!
        let prefill = ref["in.prefill"]!.item(Int.self), n = text.dim(0)
        var errors: [String: Double] = [:]

        let cache = model.newCache()
        var hidden: [MLXArray] = []
        var spans = [(0, prefill)]
        spans += (prefill ..< n).map { ($0, $0 + 1) }
        for (index, (s, e)) in spans.enumerated() {
            let emb = model.embed(tokens: text[s ..< e].expandedDimensions(axis: 0),
                                  modelLatents: modelLatents[s ..< e].expandedDimensions(axis: 0),
                                  userLatents: userLatents[s ..< e].expandedDimensions(axis: 0),
                                  modelMask: modelMask[s ..< e].expandedDimensions(axis: 0),
                                  userMask: userMask[s ..< e].expandedDimensions(axis: 0))
            if index == 0 {
                errors["embed"] = Parity.relativeError(emb[0], ref["lm.embed"]!)
                var h = emb
                for (i, layer) in model.lm.layers.enumerated() {
                    h = layer(h, index: i, cache)
                    let e = Parity.relativeError(h[0], ref["lm.layer.\(i)"]!)
                    if [0, 3, 15, 31].contains(i) { errors["layer.\(i)"] = e }
                }
                hidden.append(model.lm.norm(h)[0])
            } else {
                hidden.append(model.forward(emb, cache)[0])
            }
            eval(hidden.last!, cache.arrays)
        }
        let all = concatenated(hidden, axis: 0)
        errors["hidden.prefill"] = Parity.relativeError(all[..<prefill], ref["lm.hidden"]![..<prefill])
        errors["hidden.steps"] = Parity.relativeError(all[prefill...], ref["lm.hidden"]![prefill...])
        let logits = model.logits(all[prefill...])
        errors["logits"] = Parity.relativeError(logits, ref["lm.logits"]!)
        let refTop = argMax(ref["lm.logits"]!, axis: -1), top = argMax(logits, axis: -1)
        errors["top1.mismatches"] = Double((refTop .!= top).sum().item(Int.self))
        let stream = MLXArray([Int32(model.config.padTokenId), Int32(model.config.eopTokenId)])
        errors["logits.stream_fp32"] = Parity.relativeError(model.logitsFP32(all[prefill...], tokens: stream),
                                                            ref["lm.logits_stream_fp32"]!)
        let latents = model.head.sample(all, noise: ref["in.noise"]!.asType(model.dtype), guidance: model.config.head.guidanceScale)
        errors["head"] = Parity.relativeError(latents, ref["head.latents"]!)
        print("MODEL-PARITY \(name): " + errors.sorted { $0.key < $1.key }.map { "\($0.key)=\(String(format: "%.2e", $0.value))" }.joined(separator: " "))
        return errors
    }

    @Test func float32() throws {
        // Measured 2026-11-17: hidden 1.4e-5 (prompt) / 1.7e-6 (steps), logits 1.4e-6, PAD/EOP fp32 6e-7,
        // head 5.6e-6 — upstream's levels against torch (9e-6 / 6e-6 / 2e-6).
        let e = try run(.float32, "float32")
        #expect(e["top1.mismatches"] == 0)
        for key in ["embed", "hidden.prefill", "hidden.steps", "logits", "logits.stream_fp32", "head"] {
            #expect(e[key]! < 5e-5, "\(key) \(e[key]!)")
        }
    }

    @Test func bfloat16() throws {
        // Not bit-exact: MLX Swift and MLX Python build their Metal kernels differently (rms_norm differs
        // by 3.5e-8 in fp32 on identical inputs), which flips bf16 roundings that then accumulate
        // (docs/knowledge/pitfalls/metal-kernels-differ-between-builds.md). Measured 2026-11-17: hidden
        // 4.7e-2 (prompt) / 1.3e-2 (steps), logits 1.6e-2, head 2.9e-2; upstream's bf16-vs-fp32 noise is
        // 6e-2 to 9e-2. The gate is the same top-1 token and errors inside that noise.
        let e = try run(.bfloat16, "bfloat16")
        #expect(e["top1.mismatches"] == 0)
        #expect(e["embed"]! == 0)
        for key in ["hidden.prefill", "hidden.steps", "logits", "head"] {
            #expect(e[key]! < 1e-1, "\(key) \(e[key]!)")
        }
    }

    @Test func deltaKernelMatchesArrays() throws {
        let q = MLXRandom.normal([1, 5, 2, 64], key: MLXRandom.key(1))
        let k = MLXRandom.normal([1, 5, 2, 64], key: MLXRandom.key(2)) * 0.1
        let v = MLXRandom.normal([1, 5, 4, 32], key: MLXRandom.key(3))
        let g = MLXRandom.uniform(0.5 ..< 1.0, [1, 5, 4], key: MLXRandom.key(4))
        let beta = MLXRandom.uniform(0.0 ..< 1.0, [1, 5, 4], key: MLXRandom.key(5))
        let state = MLXRandom.normal([1, 4, 32, 64], key: MLXRandom.key(6))
        let (y1, s1) = GatedDeltaRule.metal(q, k, v, g, beta, state)
        let (y2, s2) = GatedDeltaRule.arrays(q, k, v, g, beta, state)
        #expect(Parity.relativeError(y1, y2) < 1e-5)
        #expect(Parity.relativeError(s1, s2) < 1e-5)
    }
}
