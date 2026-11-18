// RealSessionReplayTests.swift - a whole haru_mlx session replayed in Swift with Python's choices forced
// Copyright 2026 Vincent Gourbin
//
// Scripts/reference/dump_replay.py records a Python session frame by frame (token, latent fed back,
// hidden state, user latent, audio). Swift runs the same prompt and user audio and, at every frame,
// takes Python's token and latent instead of its own samples, so both sides see the same history for
// all ~400 frames. The error of Swift's hidden state, own head latent, user latent and audio against
// frame index then shows whether anything drifts over a long session (streaming codec, KV growth,
// latent feedback), which the 6-step and 34-frame parity checks cannot.

import Foundation
import MLX
import Testing
@testable import HaruCore

@Suite(.serialized, .enabled(if: Parity.enabled))
struct RealSessionReplayTests {
    struct Curve {
        var hidden: [Double] = [], latent: [Double] = [], user: [Double] = [], audio: [Double] = []
        /// Frames where Swift's own sampled token differs from Python's: (frame, Swift, Python).
        var tokenMismatches: [(Int, Int, Int)] = []
    }

    func replay(_ dtype: DType, _ name: String, quantize: Bool = false) async throws -> Curve? {
        let url = URL(fileURLWithPath: ProcessInfo.processInfo.environment["HARU_PARITY_DIR"]!)
            .appendingPathComponent("replay-\(name).safetensors")
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        let ref = try MLX.loadArrays(url: url)
        let model = HaruModel(try Parity.store(dtype), config: try Parity.config())
        if quantize { try model.quantize(bits: 8) }
        model.attach(try await HaruTokenizer.load(directory: Parity.checkpoint))
        let library = try CardLibrary.bundled()
        let card = try library.presets(language: "en").sorted(by: { $0.key < $1.key }).first!.value
        var config = DuplexConfig()
        config.seed = UInt64(ref["in.seed"]!.item(Int32.self))
        config.eopBias = ref["in.eop_bias"]!.item(Float.self)
        let user = ref["in.user"]!.asArray(Float.self)
        let fs = model.config.frameSamples, count = user.count / fs
        config.maxFrames = count + 1
        let session = DuplexSession(model: model, systemPrompt: try library.render(card, language: "en"),
                                    reference: ref["in.reference"]!.asArray(Float.self), config: config)
        let tokens = ref["tokens"]!.asArray(Int32.self), latents = ref["latents"]!
        let hidden = ref["hidden"]!, userLatents = ref["user_latents"]!, audio = ref["audio"]!
        var curve = Curve()
        var frame = 0
        session.forced = { i in (Int(tokens[i]), latents[i]) }
        session.onTrace = { h, l, t in
            if t != Int(tokens[frame]) { curve.tokenMismatches.append((frame, t, Int(tokens[frame]))) }
            curve.hidden.append(Parity.relativeError(h.reshaped([-1]), hidden[frame]))
            curve.latent.append(Parity.relativeError(l.reshaped([-1]), latents[frame]))
        }
        session.onUserLatent = { z in curve.user.append(Parity.relativeError(z.reshaped([-1]), userLatents[frame - 1])) }
        var frames = [session.start()]
        frame = 1
        for i in 0 ..< (count - 1) {
            frames.append(try session.step(Array(user[(i * fs) ..< ((i + 1) * fs)])))
            frame += 1
        }
        for (i, f) in frames.enumerated() {
            curve.audio.append(Parity.relativeError(MLXArray(f.audio), audio[i]))
        }
        return curve
    }

    func report(_ name: String, _ c: Curve) {
        func window(_ v: [Double], _ r: Range<Int>) -> String {
            let w = v[max(0, r.lowerBound) ..< min(v.count, r.upperBound)].sorted()
            return w.isEmpty ? "-" : String(format: "%.1e/%.1e", w[w.count / 2], w.last!)
        }
        let n = c.hidden.count
        print("REPLAY \(name) own token differs from Python's at \(c.tokenMismatches.count)/\(n) frames: " +
              c.tokenMismatches.prefix(20).map { "\($0.0):\($0.1)/\($0.2)" }.joined(separator: " "))
        for r in stride(from: 0, to: n, by: 50).map({ $0 ..< $0 + 50 }) {
            print("REPLAY \(name) frames \(r.lowerBound)-\(min(n, r.upperBound) - 1): hidden \(window(c.hidden, r)) " +
                  "latent \(window(c.latent, r)) user \(window(c.user, r)) audio \(window(c.audio, r)) (median/max)")
        }
    }

    @Test func float32() async throws {
        guard let c = try await replay(.float32, "float32") else { return }
        report("float32", c)
        // Same history on both sides: errors must stay at the fp32 parity level for the whole session.
        #expect(c.hidden.max()! < 1e-3)
        #expect(c.latent.max()! < 1e-2)
        #expect(c.user.max()! < 1e-3)
        // Sampling (DRY, top-k, fp32 PAD/EOP) with the same history and key: same token at (almost) every frame.
        #expect(c.tokenMismatches.count <= 2, "\(c.tokenMismatches)")
    }

    @Test func bfloat16() async throws {
        guard let c = try await replay(.bfloat16, "bfloat16") else { return }
        report("bfloat16", c)
        // Gate: no growth over the session (late median within 2x of the early median).
        let early = c.hidden[1 ..< 60].sorted()[29], late = c.hidden[(c.hidden.count - 60)...].sorted()[29]
        #expect(late < 2 * early + 1e-3, "hidden error grows: \(early) -> \(late)")
    }

    /// bf16 and 8-bit against the fp32 truth (Python's fp32 session forced): how far each precision
    /// is from fp32, frame by frame. The 8-bit profile is admissible if its distance stays close to bf16's.
    @Test func precisionsAgainstFloat32() async throws {
        guard let bf16 = try await replay(.bfloat16, "float32"),
              let int8 = try await replay(.bfloat16, "float32", quantize: true) else { return }
        report("bf16-vs-fp32", bf16)
        report("int8-vs-fp32", int8)
        func median(_ v: [Double]) -> Double { v.sorted()[v.count / 2] }
        print(String(format: "PRECISION vs fp32 (median over the session): bf16 hidden %.2e latent %.2e audio %.2e · int8 hidden %.2e latent %.2e audio %.2e",
                     median(bf16.hidden), median(bf16.latent), median(bf16.audio), median(int8.hidden), median(int8.latent), median(int8.audio)))
    }
}
