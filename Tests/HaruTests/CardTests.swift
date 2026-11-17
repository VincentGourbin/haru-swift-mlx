// CardTests.swift - card rendering against haru_mlx, and token ids against its tokenizer
// Copyright 2026 Vincent Gourbin
//
// Fixtures/cards.json is written by Scripts/reference/dump_cards.py (haru_mlx @ 7fe6b76).

import Foundation
import Testing
@testable import HaruCore

struct CardFixture: Decodable {
    struct Card: Decodable { let language: String; let preset: String; let text: String; let ids: [Int] }
    struct Extra: Decodable { let text: String; let ids: [Int] }
    let cards: [Card]
    let extra: [Extra]

    static func load() throws -> CardFixture {
        let url = try #require(Bundle.module.url(forResource: "cards", withExtension: "json", subdirectory: "Fixtures"))
        return try JSONDecoder().decode(CardFixture.self, from: Data(contentsOf: url))
    }
}

@Suite struct CardTests {
    @Test func everyExampleCardRendersAsUpstream() throws {
        let fixture = try CardFixture.load()
        let library = try CardLibrary.bundled()
        #expect(library.languages.count == 9)
        #expect(fixture.cards.count == 9)
        for expected in fixture.cards {
            let card = try #require(try library.presets(language: expected.language)[expected.preset])
            #expect(try library.render(card, language: expected.language) == expected.text, "\(expected.language)")
        }
    }

    @Test func emptyAndUnknownOptionsAreDropped() throws {
        let template = try #require(try CardLibrary.bundled().templates["en"])
        let text = template.render(CharacterCard(name: "  Haru \n", gender: "other", backchannel: "high", situation: ""))
        #expect(text == "\(template.lead)\n[Your Name]\nHaru\n[Your Backchannels]\nfrequent")
    }

    @Test func unknownLanguageIsRejected() throws {
        #expect(throws: HaruError.self) { try CardLibrary.bundled().render(CharacterCard(), language: "it") }
    }
}

/// Token ids against the checkpoint's tokenizer (gated on HARU_MODELS_DIR).
@Suite(.enabled(if: ProcessInfo.processInfo.environment["HARU_MODELS_DIR"] != nil))
struct RealTokenizerTests {
    @Test func systemPromptIdsMatchUpstream() async throws {
        let fixture = try CardFixture.load()
        let directory = ModelsDirectory.resolve().appendingPathComponent(HaruReference.repository)
        let tokenizer = try await HaruTokenizer.load(directory: directory)
        for card in fixture.cards {
            #expect(tokenizer.systemPromptIds(card.text) == card.ids, "card \(card.language)")
        }
        for extra in fixture.extra {
            #expect(tokenizer.systemPromptIds(extra.text) == extra.ids, "\(extra.text.debugDescription)")
        }
    }

    @Test func decodeRoundTripsAndKeepsSpecialTokens() async throws {
        let directory = ModelsDirectory.resolve().appendingPathComponent(HaruReference.repository)
        let tokenizer = try await HaruTokenizer.load(directory: directory)
        let text = "Tu t'appelles Haru : « déjà vu » 🙂"
        let ids = tokenizer.systemPromptIds(text)
        #expect(tokenizer.decode(ids) == "<|im_start|>system\n\(text)<|im_end|>\n")
    }
}
