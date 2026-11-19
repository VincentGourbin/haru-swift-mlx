// RunCommand.swift - `haru run`: a recording of the user in, the model's side of the conversation out
// Copyright 2026 Vincent Gourbin
// Mirrors haru_mlx/run_inference.py (higurashi-ai/Haru @ 7fe6b76).

import ArgumentParser
import Foundation
import HaruCore
import MLX

enum Precision: String, ExpressibleByArgument, CaseIterable {
    case bf16, fp32, int8
}

struct ModelOptions: ParsableArguments {
    @Option(help: "Checkpoint directory (default: $HARU_MODELS_DIR/higurashi-ai/Haru-4B).")
    var model: String?

    @Option(help: "Precision: bf16 (default, as published), fp32 (parity), int8 (8-bit backbone).")
    var precision: Precision = .bf16

    func load() async throws -> HaruModel {
        let directory = model.map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath, isDirectory: true) }
            ?? HaruWeights.directory()
        Memory.cacheLimit = 1 << 30  // as upstream's LocalBackend: frames reuse small buffers
        return try await HaruModel.load(directory: directory, dtype: precision == .fp32 ? .float32 : nil,
                                        quantize: precision == .int8 ? 8 : nil)
    }
}

struct CardOptions: ParsableArguments {
    @Option(help: "Language of the character card (ja, en, zh, ko, es, fr, de, pt, ru).")
    var language = "en"

    @Option(help: "Card JSON file; default: the language's example card.")
    var cardFile: String?

    @Option(help: "Raw system prompt instead of a card.")
    var systemPrompt: String?

    @Option(help: "Voice reference (3 s to 3 min of one speaker); default: $HARU_TEST_VOICE.")
    var voice: String?

    func systemPromptText() throws -> String {
        if let systemPrompt { return systemPrompt }
        let library = try CardLibrary.bundled()
        let card: CharacterCard
        if let cardFile {
            card = try JSONDecoder().decode(CharacterCard.self, from: Data(contentsOf: URL(fileURLWithPath: (cardFile as NSString).expandingTildeInPath)))
        } else {
            guard let preset = try library.presets(language: language).sorted(by: { $0.key < $1.key }).first?.value else {
                throw HaruError.invalidRequest("no example card for \(language)")
            }
            card = preset
        }
        return try library.render(card, language: language)
    }

    func referenceAudio() throws -> [Float]? {
        guard let path = voice ?? ProcessInfo.processInfo.environment["HARU_TEST_VOICE"] else { return nil }
        return AudioFile.prepareReference(try AudioFile.load(URL(fileURLWithPath: (path as NSString).expandingTildeInPath)))
    }
}

struct Run: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Offline conversation: the user's recording in, the model's audio (24 kHz WAV) out.")

    @Argument(help: "Audio of the user (any format AVFoundation reads; mixed down to mono).") var input: String
    @Argument(help: "Audio of the model (24 kHz WAV).") var output: String
    @OptionGroup var modelOptions: ModelOptions
    @OptionGroup var card: CardOptions
    @Option(help: "Audio guidance scale (1-2).") var guidanceScale: Float = 1.5
    @Option(help: "Text sampling temperature.") var temperature: Float = 0.7
    @Option(help: "Text sampling top-k.") var topK = 25
    @Option(help: "Sampling seed.") var seed: UInt64?
    @Option(help: "Run seeds 0..<N with one load; `{seed}` in the output and transcript paths is replaced.") var seeds: Int?
    @Option(help: "Added to the \"start speaking\" logit (> 0: Haru takes the turn sooner).") var eopBias: Float = 0
    @Flag(help: "The model opens the conversation.") var speakFirst = false
    @Flag(help: "Write the user on the left channel and the model on the right.") var stereo = false
    @Option(help: "Write per-frame text and tokens as JSON lines.") var transcript: String?
    @Option(help: "Steering: words imposed on Haru's text stream (its voice is left to the model).") var steer: String?
    @Option(help: "Steering starts after this time in seconds.") var steerAfter: Double = 0
    @Option(help: "eop: when Haru itself starts speaking; now: Haru is made to start at --steer-after.") var steerWhen = "eop"
    @Option(help: "Seconds of forced silence (text) after the imposed words.") var steerHold: Double = 3
    @Option(help: "fixed: one imposed token per frame; model: only the frames where Haru writes a word get one (its pauses are kept).") var steerPace = "fixed"

    func run() async throws {
        let model = try await modelOptions.load()
        if let seeds {
            for s in 0 ..< seeds { try converse(model, seed: UInt64(s), tag: String(s)) }
        } else {
            try converse(model, seed: seed, tag: "")
        }
    }

    func converse(_ model: HaruModel, seed: UInt64?, tag: String) throws {
        let output = output.replacingOccurrences(of: "{seed}", with: tag)
        let transcript = transcript?.replacingOccurrences(of: "{seed}", with: tag)
        let frameSamples = model.config.frameSamples
        var user = try AudioFile.load(URL(fileURLWithPath: (input as NSString).expandingTildeInPath))
        let count = max(1, (user.count + frameSamples - 1) / frameSamples)
        user += [Float](repeating: 0, count: count * frameSamples - user.count)
        var config = DuplexConfig()
        config.guidanceScale = guidanceScale
        config.temperature = temperature
        config.topK = topK
        config.speakFirst = speakFirst
        config.maxFrames = count + 1
        config.seed = seed
        config.eopBias = eopBias
        let session = DuplexSession(model: model, systemPrompt: try card.systemPromptText(),
                                    reference: try card.referenceAudio(), config: config)
        if let steer {
            let steerer = Steerer(tokens: model.tokenizer!.encode(" " + steer), pad: model.config.padTokenId,
                                  eop: model.config.eopTokenId, frameSeconds: Double(frameSamples) / 24_000,
                                  after: steerAfter, now: steerWhen == "now", hold: steerHold, paced: steerPace == "model")
            session.textOverride = { steerer.override($0, $1) }
        }
        // Model frame t is heard while the user speaks frame t: the last user frame needs no step.
        var frames = [session.start()]
        for i in 0 ..< (count - 1) {
            frames.append(try session.step(Array(user[(i * frameSamples) ..< ((i + 1) * frameSamples)])))
        }
        let reply = frames.flatMap(\.audio)
        let out = URL(fileURLWithPath: (output as NSString).expandingTildeInPath)
        if stereo {
            try AudioFile.writeWAV(zip(user, reply).flatMap { [$0, $1] }, to: out, channels: 2)
        } else {
            try AudioFile.writeWAV(reply, to: out)
        }
        if let transcript {
            let lines = frames.map { f -> String in
                let json = try! JSONSerialization.data(withJSONObject: ["frame": f.index, "token": f.token, "text": f.text, "ms": f.seconds * 1000,
                                                                        "time": Double(f.index * frameSamples) / 24_000])
                return String(decoding: json, as: UTF8.self)
            }
            try (lines.joined(separator: "\n") + "\n").write(toFile: (transcript as NSString).expandingTildeInPath, atomically: true, encoding: .utf8)
        }
        let steps = frames.dropFirst().map { $0.seconds * 1000 }.sorted()
        func pct(_ p: Double) -> Double { steps.isEmpty ? 0 : steps[min(steps.count - 1, Int(p * Double(steps.count)))] }
        print(String(format: "%d frames (%.1f s), prompt %d positions in %.2f s, %.1f ms per frame (p99 %.1f) -> %@",
                     count, Double(count * frameSamples) / 24_000, session.promptLength, frames[0].seconds, pct(0.5), pct(0.99), output))
        print(frames.map(\.text).joined().trimmingCharacters(in: .whitespaces))
    }
}

/// Imposes a sentence on Haru's text stream once, after a given time: either from the frame where Haru
/// itself starts speaking (EOP), or by making it start; then holds the text silent for a while.
final class Steerer {
    let tokens: [Int], pad: Int, eop: Int, frameSeconds: Double, after: Double, now: Bool, holdFrames: Int, paced: Bool
    private var position: Int?
    private var holdUntil: Int?
    private var done = false

    init(tokens: [Int], pad: Int, eop: Int, frameSeconds: Double, after: Double, now: Bool, hold: Double, paced: Bool) {
        (self.tokens, self.pad, self.eop, self.frameSeconds, self.after, self.now) = (tokens, pad, eop, frameSeconds, after, now)
        self.paced = paced
        holdFrames = Int(hold / frameSeconds)
    }

    func override(_ index: Int, _ sampled: Int) -> Int? {
        if done { return nil }
        if let until = holdUntil {
            if index < until { return pad }
            done = true
            return nil
        }
        if let k = position {
            // Paced: Haru's own PAD and EOP frames are kept (its pauses and phrase starts), except that the
            // first imposed token always follows the start.
            if paced, k > 0, k < tokens.count, sampled == pad || sampled == eop { return nil }
            if k < tokens.count { position = k + 1; return tokens[k] }
            holdUntil = index + holdFrames
            return pad
        }
        guard Double(index) * frameSeconds >= after else { return nil }
        if now {
            position = 0
            print(String(format: "steer: Haru made to speak at %.2f s", Double(index) * frameSeconds))
            return eop
        }
        if sampled == eop {
            position = 0
            print(String(format: "steer: Haru started speaking at %.2f s, imposing %d tokens", Double(index) * frameSeconds, tokens.count))
        }
        return nil
    }
}
