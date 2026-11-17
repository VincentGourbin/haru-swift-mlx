// NoiseSourceTests.swift - same seed, same draws as haru_mlx (Python MLX), bit for bit
// Copyright 2026 Vincent Gourbin
//
// Fixtures/rng.json is written by Scripts/reference/dump_rng.py (mlx 0.32.4, GPU), replaying the
// order of draws of haru_mlx/session.py. Keys, bfloat16 noise and categorical draws must be
// identical; float32 normal draws within 1 ulp.

import Foundation
import MLX
import Testing
@testable import HaruCore

struct RNGFixture: Decodable {
    struct Draw: Decodable {
        let draw_key: [UInt32]
        let normal_f32: [UInt32]
        let head_bf16: [UInt16]
        let codec_scale: [UInt32]
        let codec_noise: [UInt32]
        let categorical: [Int]
    }
    struct Seed: Decodable {
        let seed: UInt64
        let key: [UInt32]
        let draws: [Draw]
        let final_key: [UInt32]
    }
    let mlx: String
    let seeds: [Seed]

    static func load() throws -> RNGFixture {
        let url = try #require(Bundle.module.url(forResource: "rng", withExtension: "json", subdirectory: "Fixtures"))
        return try JSONDecoder().decode(RNGFixture.self, from: Data(contentsOf: url))
    }
}

private func f32Bits(_ a: MLXArray) -> [UInt32] { a.asArray(Float.self).map(\.bitPattern) }
private func u32(_ a: MLXArray) -> [UInt32] { a.asArray(UInt32.self) }

/// Largest distance in units in the last place between two float32 bit patterns of the same length.
private func maxUlps(_ a: [UInt32], _ b: [UInt32]) -> Int64 {
    guard a.count == b.count else { return .max }
    return zip(a, b).map { abs(Int64(Int32(bitPattern: $0)) - Int64(Int32(bitPattern: $1))) }.max() ?? 0
}

@Suite(.serialized) struct NoiseSourceTests {
    @Test func sessionDrawsMatchPythonBitForBit() throws {
        let fixture = try RNGFixture.load()
        for expected in fixture.seeds {
            let noise = NoiseSource(seed: expected.seed)
            #expect(u32(noise.state) == expected.key, "seed \(expected.seed): initial key")
            for (step, draw) in expected.draws.enumerated() {
                let label = "seed \(expected.seed), step \(step)"
                let key = noise.nextKey()
                #expect(u32(key) == draw.draw_key, "\(label): draw key")
                #expect(f32Bits(MLXRandom.normal([1, 64], key: key)) == draw.normal_f32, "\(label): normal f32")
                let head = noise.normal([1, 64], dtype: .bfloat16)
                #expect(head.view(dtype: .uint16).asArray(UInt16.self) == draw.head_bf16, "\(label): head noise bf16")
                let (scaleKey, noiseKey) = MLXRandom.split(key: noise.nextKey())
                #expect(f32Bits(MLXRandom.normal([1, 1, 1], key: scaleKey)) == draw.codec_scale, "\(label): codec scale")
                // The keys are identical, so the random bits are; the bits -> normal transform may round
                // one element differently (measured 2026-11-17: 1 of 1152 values, 1 ulp; docs/knowledge/decisions/rng.md).
                #expect(maxUlps(f32Bits(MLXRandom.normal([1, 3, 64], key: noiseKey)), draw.codec_noise) <= 1, "\(label): codec noise")
                let logits = MLXArray((0..<25).map { Float(-2.0 + 4.0 * Double($0) / 24.0) }).reshaped([1, 25]) / 0.7
                let token = MLXRandom.categorical(logits, key: noise.nextKey())
                #expect(token.asArray(Int32.self).map(Int.init) == draw.categorical, "\(label): categorical")
            }
            #expect(u32(noise.state) == expected.final_key, "seed \(expected.seed): final key")
        }
    }

    @Test func randomSeedFitsIn32Bits() {
        for _ in 0..<100 { #expect(NoiseSource.randomSeed() <= UInt64(UInt32.max)) }
    }
}
