// WeightInventory.swift - what a Haru checkpoint holds, by part, from its safetensors header
// Copyright 2026 Vincent Gourbin
//
// A live frame reads (almost) every weight once: the backbone, the tied output embedding, the
// MeanFlow head and the deep stages of the codec are matrix-vector products. Bytes per frame are
// therefore the first-order cost of real time; this inventory is where `plan/03-budget.md` gets them.

import Foundation

public struct WeightInventory: Sendable, Equatable {
    /// The parts named in the checkpoint README (`lm.*`, `*_audio_proj.*`, `head.*`, `codec.*`).
    public enum Part: String, CaseIterable, Sendable {
        case embedding = "lm.embed_tokens"
        case backbone = "lm.layers + lm.norm"
        case audioProjectors = "model/user_audio_proj"
        case head = "head"
        case codecEncoder = "codec.encoder"
        case codecDecoder = "codec.decoder"
        case other = "other"

        static func of(_ name: String) -> Part {
            if name.hasPrefix("lm.embed_tokens.") { return .embedding }
            if name.hasPrefix("lm.") { return .backbone }
            if name.hasPrefix("model_audio_proj.") || name.hasPrefix("user_audio_proj.") { return .audioProjectors }
            if name.hasPrefix("head.") { return .head }
            if name.hasPrefix("codec.encoder.") { return .codecEncoder }
            if name.hasPrefix("codec.decoder.") { return .codecDecoder }
            return .other
        }
    }

    public struct Line: Sendable, Equatable {
        public let part: Part
        public let tensors: Int
        public let parameters: Int
        public let bytes: Int
    }

    public let lines: [Line]

    public init(header: SafetensorsHeader) {
        var tensors: [Part: Int] = [:], parameters: [Part: Int] = [:], bytes: [Part: Int] = [:]
        for (name, entry) in header.entries {
            let part = Part.of(name)
            let count = entry.shape.reduce(1, *)
            tensors[part, default: 0] += 1
            parameters[part, default: 0] += count
            bytes[part, default: 0] += count * Self.bytesPerElement(entry.dtype)
        }
        lines = Part.allCases.compactMap { part in
            guard let n = tensors[part] else { return nil }
            return Line(part: part, tensors: n, parameters: parameters[part] ?? 0, bytes: bytes[part] ?? 0)
        }
    }

    public var totalParameters: Int { lines.reduce(0) { $0 + $1.parameters } }
    public var totalBytes: Int { lines.reduce(0) { $0 + $1.bytes } }

    static func bytesPerElement(_ dtype: String) -> Int {
        switch dtype {
        case "F64", "I64", "U64": 8
        case "F32", "I32", "U32": 4
        case "BF16", "F16", "I16", "U16": 2
        default: 1
        }
    }
}
