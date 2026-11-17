// RealCodecParityTests.swift - VibeVoice codec against haru_mlx on the real weights
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import Testing
@testable import HaruCore

@Suite(.serialized, .enabled(if: Parity.enabled))
struct RealCodecParityTests {
    func check(_ dtype: DType, _ name: String, tolerance: [String: Double]) throws {
        let ref = try Parity.reference(name)
        let config = try Parity.config()
        let codec = VibeVoiceCodec(try Parity.store(dtype), config: config.codec)
        var errors: [String: Double] = [:]

        errors["encode_clip"] = Parity.relativeError(codec.encode(ref["in.clip"]!)[0], ref["codec.encode_clip"]!)
        errors["decode_latents"] = Parity.relativeError(codec.decode(ref["in.latents"]!)[0], ref["codec.decode_latents"]!)
        let stream = CodecStream(codec)
        errors["stream_encode"] = Parity.relativeError(stream.encode(ref["in.frames"]!.reshaped([1, -1]))[0], ref["codec.stream_encode"]!)
        errors["stream_decode"] = Parity.relativeError(stream.decode(ref["in.latents"]!)[0], ref["codec.stream_decode"]!)
        let fed = CodecStream(codec)
        let frames = ref["in.frames"]!
        for (s, e) in [(0, 1000), (1000, 5300), (5300, 9600)] { fed.feed(frames[s ..< e]) }
        let popped = concatenated((0..<fed.framesReady).map { _ in fed.pop() }, axis: 1)[0]
        errors["feed_encode"] = Parity.relativeError(popped, ref["codec.feed_encode"]!)

        print("CODEC-PARITY \(name): " + errors.sorted { $0.key < $1.key }.map { "\($0.key)=\(String(format: "%.2e", $0.value))" }.joined(separator: " "))
        for (key, error) in errors {
            #expect(error <= tolerance[key]!, "\(name) \(key): \(error)")
        }
    }

    @Test func float32() throws {
        // Measured 2026-11-17: encoder 4.7e-7, decoder 1.4e-5 (bf16 is bit-exact, so the fp32 gap is an
        // arithmetic difference, not a port error); thresholds at upstream's levels against torch.
        try check(.float32, "float32", tolerance: ["encode_clip": 2e-6, "decode_latents": 2e-5, "stream_encode": 2e-6,
                                                   "stream_decode": 2e-5, "feed_encode": 2e-6])
    }

    @Test func bfloat16() throws {
        // Measured 2026-11-17: bit-exact (0) on all five.
        try check(.bfloat16, "bfloat16", tolerance: ["encode_clip": 1e-6, "decode_latents": 1e-6, "stream_encode": 1e-6,
                                                     "stream_decode": 1e-6, "feed_encode": 1e-6])
    }

    @Test func mixerKernelMatchesArrays() throws {
        let x = MLXRandom.normal([40, 64], key: MLXRandom.key(1))
        let n = MLXRandom.normal([40, 64], key: MLXRandom.key(2))
        let state = MLXRandom.normal([6, 64], key: MLXRandom.key(3))
        let w = MLXRandom.normal([64, 7], key: MLXRandom.key(4))
        let b = MLXRandom.normal([64], key: MLXRandom.key(5))
        let g = MLXRandom.normal([64], key: MLXRandom.key(6))
        let (k1, s1) = DepthwiseMixer.metal(x, n, state, w, b, g)
        let (a1, s2) = DepthwiseMixer.arrays(x, n, state, w, b, g)
        #expect(Parity.relativeError(k1, a1) < 1e-6)
        #expect(Parity.relativeError(s1, s2) == 0)
    }
}
