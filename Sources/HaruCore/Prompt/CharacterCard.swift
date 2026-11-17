// CharacterCard.swift - the character card (system prompt) and its per-language templates
// Copyright 2026 Vincent Gourbin
//
// Port of `render_card` in haru_mlx/prompt.py (higurashi-ai/Haru @ 7fe6b76, Apache-2.0). The
// templates and the example cards are upstream's `assets/card_templates.json` and
// `assets/cards/<language>.json`, bundled as `Resources/Cards/card_templates.json` and
// `card_<language>.json`.

import Foundation

/// The fields of a card, in the order they are rendered (`CARD_FIELDS`).
public struct CharacterCard: Codable, Sendable, Equatable {
    public var persona: String?
    public var name: String?
    /// `female` or `male` (a key of the template's `gender` table).
    public var gender: String?
    /// `low`, `mid` or `high` (a key of the template's `backchannel` table).
    public var backchannel: String?
    public var otherName: String?
    public var relationship: String?
    public var situation: String?

    enum CodingKeys: String, CodingKey {
        case persona, name, gender, backchannel, otherName = "other_name", relationship, situation
    }

    public init(persona: String? = nil, name: String? = nil, gender: String? = nil, backchannel: String? = nil,
                otherName: String? = nil, relationship: String? = nil, situation: String? = nil) {
        self.persona = persona
        self.name = name
        self.gender = gender
        self.backchannel = backchannel
        self.otherName = otherName
        self.relationship = relationship
        self.situation = situation
    }

    /// `(key, value)` in rendering order.
    var fields: [(String, String?)] {
        [("persona", persona), ("name", name), ("gender", gender), ("backchannel", backchannel),
         ("other_name", otherName), ("relationship", relationship), ("situation", situation)]
    }
}

/// One language's wording: lead sentence, field headers, option labels.
public struct CardTemplate: Codable, Sendable, Equatable {
    public let lead: String
    public let headers: [String: String]
    public let gender: [String: String]
    public let backchannel: [String: String]

    /// The card text: the lead, then header and value of every non-empty field, one per line.
    public func render(_ card: CharacterCard) -> String {
        var lines = [lead]
        for (key, raw) in card.fields {
            var value = (raw ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
            if key == "gender" { value = gender[value] ?? "" }
            if key == "backchannel" { value = backchannel[value] ?? "" }
            if !value.isEmpty, let header = headers[key] { lines += [header, value] }
        }
        return lines.joined(separator: "\n")
    }
}

/// The bundled templates and example cards of the nine languages.
public struct CardLibrary: Sendable {
    public struct Language: Codable, Sendable, Equatable {
        public let code: String
        public let label: String
    }

    public let languages: [Language]
    public let templates: [String: CardTemplate]

    struct File: Codable {
        let languages: [Language]
        let templates: [String: CardTemplate]
    }

    public static func bundled() throws -> CardLibrary {
        let file: File = try decodeResource("card_templates")
        return CardLibrary(languages: file.languages, templates: file.templates)
    }

    /// The example cards of a language (`chat` for every language upstream), by preset id.
    public func presets(language: String) throws -> [String: CharacterCard] {
        try Self.decodeResource("card_\(language)")
    }

    public func render(_ card: CharacterCard, language: String) throws -> String {
        guard let template = templates[language] else {
            throw HaruError.invalidRequest("language must be one of \(templates.keys.sorted())")
        }
        return template.render(card)
    }

    static func decodeResource<T: Decodable>(_ name: String) throws -> T {
        guard let url = Bundle.module.url(forResource: name, withExtension: "json", subdirectory: "Cards")
                ?? Bundle.module.url(forResource: name, withExtension: "json"),
              let data = FileManager.default.contents(atPath: url.path)
        else { throw HaruError.missingFile("\(name).json (bundle)") }
        return try JSONDecoder().decode(T.self, from: data)
    }
}
