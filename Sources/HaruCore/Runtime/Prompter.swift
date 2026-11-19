// Prompter.swift - lines imposed on Haru's text stream: someone else chooses the words, Haru says them
// Copyright 2026 Vincent Gourbin
//
// Haru writes its text a little ahead of its voice, one token per 133 ms frame, and its voice follows the
// text it wrote. Replacing the sampled text tokens (DuplexSession.textOverride) therefore makes Haru say
// a given sentence in its own voice, 1 to 2 frames later (docs/knowledge/investigations/souffleur-forcage-texte.md).
// A Prompter holds a queue of such lines. Each one starts no earlier than its time and only once the user
// is silent: the start-speaking token (EOP), then the line's tokens, then a few seconds of forced silence
// before Haru is free again. If the voice stays mute after the EOP, the line is started over; if the user
// starts talking during it, the rest is dropped and Haru listens.

import Foundation

public final class Prompter: @unchecked Sendable {
    public struct Settings: Sendable {
        /// Forced text silence after a line, before Haru writes freely again.
        public var holdSeconds = 3.0
        /// Output below this level (dBFS) counts as mute.
        public var muteDB: Float = -50
        /// Frames after the EOP that must all be mute to start the line over (8 = 1.07 s).
        public var muteFrames = 8
        /// Start-overs per line when the voice stays mute.
        public var retries = 1
        /// Frames of text silence before a start-over (a mute start depends on the moment: 0.67 s later
        /// it usually speaks).
        public var retryPauseFrames = 5
        /// Audio guidance during a start-over (nil: unchanged). Guidance 1 (no classifier-free guidance) was tried
        /// on a mute start and did not help: off by default.
        public var retryGuidance: Float? = nil
        /// User level (dBFS) above which the user is speaking.
        public var userDB: Float = -40
        /// Consecutive user frames above `userDB` that interrupt a line (2 = 0.27 s).
        public var interruptFrames = 2
        /// Frames of a line during which the user is not listened to (its own start).
        public var graceFrames = 3
        /// Wait for Haru's own EOP instead of making it speak at the line's time.
        public var waitForTurn = false
        /// Wait at most this many frames for Haru's own EOP, then make it speak (0: at once).
        public var turnWaitFrames = 0
        /// Haru says only the prompter's lines: between them its text is held on PAD. Measured (5 seeds,
        /// 25 lines): English 25/25 lines said against 24/25, French 25/25 against 9/25, and no mute start.
        /// Haru's own improvised answers were what made lines fail.
        public var exclusive = true
        /// Write a line phrase by phrase (split at punctuation, at most `chunkTokens` tokens), each one
        /// opened by an EOP and followed by PAD frames sized on its spoken length, as Haru's own text
        /// looks. Measured worse: the voice tends to stop after the first pause (English, 5 seeds: 18/25
        /// lines said against 24/25 in one go). Off by default.
        public var chunked = false
        public var chunkTokens = 12
        /// Speaking time per character, to size the pauses (French TTS-like pace: about 14 characters/s).
        public var secondsPerCharacter = 0.07
        public init() {}
    }

    public struct Line: Sendable {
        public let text: String
        /// One token per frame from the first one after the opening EOP: the words, the PAD pauses
        /// between phrases and the EOP that opens each next phrase.
        public let tokens: [Int]
        public let notBefore: Double
    }

    private enum State {
        case idle
        case opening(Line, attempt: Int, at: Int)   // the EOP goes at frame `at`, PAD until then
        case speaking(Line, attempt: Int, next: Int, since: Int, mute: Int)
        case holding(until: Int)
        case listening                               // interrupted: PAD while the user speaks
    }

    public private(set) var settings: Settings
    public let frameSeconds: Double
    private let pad: Int, eop: Int
    private let encode: (String) -> [Int]
    private let decode: (Int) -> String
    private let lock = NSLock()
    private var queue: [Line] = []
    private var state = State.idle
    private var userRun = 0, userQuiet = 0, waited = 0
    /// What happened, for logs and reports: (frame index, message).
    public var onEvent: (@Sendable (Int, String) -> Void)?

    public init(tokenizer: HaruTokenizer, config: HaruConfiguration, settings: Settings = Settings()) {
        self.settings = settings
        pad = config.padTokenId
        eop = config.eopTokenId
        frameSeconds = Double(config.frameSamples) / Double(config.sampleRate)
        encode = { tokenizer.encode(" " + $0) }
        decode = { tokenizer.decode([$0]) }
    }

    /// The per-frame schedule of a line (see `Settings.chunked`).
    func schedule(_ text: String) -> [Int] {
        let ids = encode(text)
        guard settings.chunked else { return ids }
        var chunks: [[Int]] = [[]]
        for id in ids {
            let piece = decode(id)
            if chunks[chunks.count - 1].count >= settings.chunkTokens, piece.hasPrefix(" ") { chunks.append([]) }
            chunks[chunks.count - 1].append(id)
            if let last = piece.trimmingCharacters(in: .whitespaces).last, ",.?!;:".contains(last) { chunks.append([]) }
        }
        chunks.removeAll { $0.isEmpty }
        var out: [Int] = []
        for (n, chunk) in chunks.enumerated() {
            if n > 0 { out.append(eop) }
            out += chunk
            guard n + 1 < chunks.count else { break }
            let characters = chunk.map(decode).joined().count
            let spoken = Double(characters) * settings.secondsPerCharacter
            let written = Double(chunk.count + 1) * frameSeconds
            out += Array(repeating: pad, count: max(1, Int((spoken - written) / frameSeconds + 0.5)))
        }
        return out
    }

    /// Queues a line, said once the time is reached and the user is silent.
    public func say(_ text: String, notBefore: Double = 0) {
        let line = Line(text: text, tokens: schedule(text), notBefore: notBefore)
        lock.withLock { queue.append(line) }
    }

    public var pending: Int { lock.withLock { queue.count } }

    /// The user's speech threshold follows the microphone (TurnDetector.adaptiveThreshold) during a session.
    public func setUserThreshold(_ db: Float) { lock.withLock { settings.userDB = db } }

    /// True while a line is being said (or its silence held).
    public var busy: Bool {
        lock.withLock { if case .idle = state { return !queue.isEmpty } else { return true } }
    }

    /// DuplexSession.textOverride: the token for frame `index`, or nil to keep Haru's own.
    public func override(_ index: Int, _ sampled: Int) -> Int? {
        lock.withLock {
            switch state {
            case .idle:
                // Exclusive: Haru says only the prompter's lines; its own words become silence.
                let own: Int? = settings.exclusive && sampled != pad ? pad : nil
                guard let line = queue.first, Double(index) * frameSeconds >= line.notBefore, userRun == 0 else {
                    waited = 0
                    return own
                }
                if settings.waitForTurn || settings.turnWaitFrames > 0 {
                    let limit = settings.waitForTurn ? Int.max : settings.turnWaitFrames
                    if sampled != eop, waited < limit { waited += 1; return own }
                    queue.removeFirst()
                    state = .speaking(line, attempt: 0, next: 0, since: index, mute: 0)
                    let took = sampled == eop
                    event(index, (took ? "Haru takes the turn" : "made to speak after \(waited) frames") + "; saying \"\(line.text)\"")
                    waited = 0
                    return took ? nil : eop
                }
                queue.removeFirst()
                state = .speaking(line, attempt: 0, next: 0, since: index, mute: 0)
                event(index, "saying \"\(line.text)\"")
                return eop
            case let .opening(line, attempt, at):
                if index < at { return pad }
                state = .speaking(line, attempt: attempt, next: 0, since: index, mute: 0)
                return eop
            case let .speaking(line, attempt, next, since, mute):
                if next < line.tokens.count {
                    state = .speaking(line, attempt: attempt, next: next + 1, since: since, mute: mute)
                    return line.tokens[next]
                }
                state = .holding(until: index + Int(settings.holdSeconds / frameSeconds))
                return pad
            case let .holding(until):
                if index < until { return pad }
                state = .idle
                return settings.exclusive && sampled != pad ? pad : nil
            case .listening:
                if userQuiet < 4 { return pad }
                state = .idle
                return settings.exclusive && sampled != pad ? pad : nil
            }
        }
    }

    /// DuplexSession.guidanceOverride: the guidance of a start-over, while it lasts.
    public func guidance(_ index: Int) -> Float? {
        lock.withLock {
            switch state {
            case let .opening(_, attempt, _) where attempt > 0: return settings.retryGuidance
            case let .speaking(_, attempt, _, _, _) where attempt > 0: return settings.retryGuidance
            default: return nil
            }
        }
    }

    /// After each frame: Haru's audio of that frame and the user's level (dBFS) of the frame it heard.
    public func observe(index: Int, output: [Float], userLevel: Float) {
        let level = TurnDetector.level(output[...])
        lock.withLock {
            if userLevel > settings.userDB { userRun += 1; userQuiet = 0 } else { userRun = 0; userQuiet += 1 }
            guard case let .speaking(line, attempt, next, since, mute) = state else { return }
            let age = index - since
            if age >= settings.graceFrames, userRun >= settings.interruptFrames {
                state = .listening
                event(index, "the user speaks: line dropped after \(next)/\(line.tokens.count) tokens")
                return
            }
            let m = level < settings.muteDB ? mute + 1 : 0
            if age < settings.muteFrames {
                state = .speaking(line, attempt: attempt, next: next, since: since, mute: m)
            } else if age == settings.muteFrames, m > settings.muteFrames, attempt < settings.retries {
                state = .opening(line, attempt: attempt + 1, at: index + 1 + settings.retryPauseFrames)
                event(index, "voice mute \(settings.muteFrames) frames after the start: starting the line over")
            } else {
                state = .speaking(line, attempt: attempt, next: next, since: since, mute: m)
            }
        }
    }

    private func event(_ index: Int, _ message: String) {
        onEvent?(index, String(format: "%.2f s ", Double(index) * frameSeconds) + message)
    }
}
