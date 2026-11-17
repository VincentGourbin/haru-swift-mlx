// HaruTokenizer.swift - the checkpoint's text tokenizer and the system-prompt ids
// Copyright 2026 Vincent Gourbin
//
// The checkpoint ships a Qwen2 BPE `tokenizer.json`; upstream loads it with the `tokenizers`
// library and encodes without added special tokens (`system_prompt_ids` in haru_mlx/prompt.py).

import Foundation
import Tokenizers

public struct HaruTokenizer: Sendable {
    let tokenizer: any Tokenizer

    /// Loads `tokenizer.json` and `tokenizer_config.json` from a checkpoint directory.
    public static func load(directory: URL) async throws -> HaruTokenizer {
        for file in ["tokenizer.json", "tokenizer_config.json"]
        where !FileManager.default.fileExists(atPath: directory.appendingPathComponent(file).path) {
            throw HaruError.missingFile(file)
        }
        return HaruTokenizer(tokenizer: try await AutoTokenizer.from(modelFolder: directory))
    }

    /// The ChatML system turn of a card; no positions at all for an empty prompt.
    public func systemPromptIds(_ systemPrompt: String) -> [Int] {
        guard !systemPrompt.isEmpty else { return [] }
        return encode("<|im_start|>system\n\(systemPrompt)<|im_end|>\n")
    }

    public func encode(_ text: String) -> [Int] {
        tokenizer.encode(text: text, addSpecialTokens: false)
    }

    /// Special tokens are kept, as `decode(..., skip_special_tokens=False)` upstream.
    public func decode(_ ids: [Int]) -> String {
        tokenizer.decode(tokens: ids, skipSpecialTokens: false)
    }
}
