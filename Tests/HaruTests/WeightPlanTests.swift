// WeightPlanTests.swift - the conversion plan of haru_mlx/weights.py, weight-free and on the real header
// Copyright 2026 Vincent Gourbin

import Foundation
import Testing
@testable import HaruCore

@Suite struct WeightPlanTests {
    @Test func fusesProjectionsInRowOrder() throws {
        let plan = try WeightPlan.conversionPlan(shapes: [
            "lm.layers.0.linear_attn.in_proj_a.weight": [4, 8],
            "lm.layers.0.linear_attn.in_proj_qkv.weight": [24, 8],
            "lm.layers.0.linear_attn.in_proj_b.weight": [4, 8],
            "lm.layers.0.linear_attn.in_proj_z.weight": [16, 8],
            "lm.layers.3.self_attn.v_proj.weight": [2, 8],
            "lm.layers.3.self_attn.q_proj.weight": [12, 8],
            "lm.layers.3.self_attn.k_proj.weight": [2, 8],
            "head.final.ada.weight": [2, 8],
            "head.layers.1.ada.weight": [3, 8],
            "head.layers.0.ada.weight": [3, 8],
        ])
        let inProj = try #require(plan["lm.layers.0.linear_attn.in_proj.weight"])
        #expect(inProj.sources == ["qkv", "z", "b", "a"].map { "lm.layers.0.linear_attn.in_proj_\($0).weight" })
        #expect(inProj.shape == [48, 8])
        #expect(plan["lm.layers.3.self_attn.qkv.weight"]?.sources.last == "lm.layers.3.self_attn.v_proj.weight")
        let ada = try #require(plan["head.ada.weight"])
        #expect(ada.sources == ["head.layers.0.ada.weight", "head.layers.1.ada.weight", "head.final.ada.weight"])
        #expect(ada.shape == [8, 8])
    }

    @Test func layoutsAndShapes() throws {
        let plan = try WeightPlan.conversionPlan(shapes: [
            "lm.norm.weight": [8],
            "lm.layers.2.self_attn.q_norm.weight": [4],
            "lm.layers.0.linear_attn.norm.weight": [4],
            "lm.layers.0.linear_attn.conv1d.weight": [24, 1, 4],
            "codec.encoder.down.1.weight": [16, 8, 4],
            "codec.decoder.stages.0.2.conv.weight": [32, 1, 7],
            "codec.decoder.up.0.weight": [64, 32, 16],
            "latent_scale": [],
            "user_audio_proj.norm.weight": [8],
        ])
        #expect(plan["lm.norm.weight"]?.layout == .gamma)
        #expect(plan["lm.layers.2.self_attn.q_norm.weight"]?.layout == .gamma)
        // GatedNorm and projector gains are not zero-centred (pitfall 2).
        #expect(plan["lm.layers.0.linear_attn.norm.weight"]?.layout == .copy)
        #expect(plan["user_audio_proj.norm.weight"]?.layout == .copy)
        #expect(plan["lm.layers.0.linear_attn.conv1d.weight"]?.shape == [24, 4, 1])
        #expect(plan["codec.encoder.down.1.weight"]?.shape == [16, 32])
        #expect(plan["codec.decoder.stages.0.2.conv.weight"]?.shape == [32, 7])
        #expect(plan["codec.decoder.up.0.weight"]?.layout == .convTransposeMatrix)
        #expect(plan["codec.decoder.up.0.weight"]?.shape == [512, 64])
        #expect(plan["latent_scale"]?.layout == .scalar)
        #expect(plan["latent_scale"]?.shape == [])
    }

    @Test func checkListsEveryKindOfProblem() throws {
        let plan = try WeightPlan.conversionPlan(shapes: ["a.weight": [2, 2], "b.weight": [3]])
        #expect(throws: HaruError.self) {
            try WeightPlan.check(plan, expected: ["a.weight": [2, 3], "c.weight": [1]])
        }
        try WeightPlan.check(plan, expected: ["a.weight": [2, 2], "b.weight": [3]])
    }

    @Test func publishedConfigurationHasTheUpstreamParameterCount() throws {
        let config = try HaruConfiguration.load(from: Data(ConfigurationTests.published.utf8))
        let shapes = WeightPlan.parameterShapes(config: config)
        // 1022 tensors: q/k/v → qkv on 8 layers (−16), 4 → in_proj on 24 (−72), gate/up on 32 (−32), 5 ada → 1 (−4).
        #expect(shapes.count == 1022 - 16 - 72 - 32 - 4)
    }
}

/// Against the downloaded checkpoint (gated on HARU_MODELS_DIR).
@Suite(.enabled(if: ProcessInfo.processInfo.environment["HARU_MODELS_DIR"] != nil))
struct RealWeightPlanTests {
    @Test func realHeaderMatchesTheModel() throws {
        let directory = ModelsDirectory.resolve().appendingPathComponent(HaruReference.repository)
        let config = try HaruConfiguration.load(contentsOf: directory.appendingPathComponent("config.json"))
        let header = try SafetensorsHeader(contentsOf: directory.appendingPathComponent("model.safetensors"))
        #expect(header.entries.count == 1022)
        let plan = try WeightPlan.conversionPlan(shapes: header.entries.mapValues(\.shape))
        try WeightPlan.check(plan, expected: WeightPlan.parameterShapes(config: config))
        #expect(plan.values.reduce(0) { $0 + $1.sources.count } == 1022)
    }
}
