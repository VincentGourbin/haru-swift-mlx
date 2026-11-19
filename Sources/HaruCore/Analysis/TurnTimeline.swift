// TurnTimeline.swift - who speaks when: speech segments per side and the reply delay of each turn
// Copyright 2026 Vincent Gourbin
//
// One level per 133 ms frame on each side (the user's microphone, Haru's output), a threshold with a
// hangover gives speech segments, and each user utterance is paired with the Haru segment that answers
// it. The delay of a turn is the time from the end of the user's utterance to the start of the reply
// (negative when Haru starts before the user has finished). Used by the app's timeline, by
// `haru turns` on a stereo WAV, and to compare the turn timing of the port with haru_mlx.

import Foundation

public struct SpeechSegment: Sendable, Equatable, Codable {
    public var start: Double  // seconds from the start of the conversation
    public var end: Double
    public init(start: Double, end: Double) { self.start = start; self.end = end }
    public var duration: Double { end - start }
}

public struct Turn: Sendable, Equatable, Codable {
    public var user: SpeechSegment
    public var reply: SpeechSegment?
    /// `reply.start - user.end`; nil when Haru did not answer before the next utterance.
    public var delay: Double? { reply.map { $0.start - user.end } }
    /// What Haru said during the reply (text tokens from a little before it starts to its end).
    public var text: String
}

public struct TurnDetector: Sendable {
    /// Frame level above which a side is speaking (dBFS of the frame's RMS).
    public var thresholdDB: Float = -40
    /// Frames of silence that end a segment (5 = 0.67 s: pauses inside a sentence stay in it).
    public var hangFrames = 5
    /// Segments shorter than this many frames are dropped (clicks, breaths).
    public var minFrames = 2
    /// Text is written ahead of the audio: tokens this long before a reply belong to it.
    public var textLead = 1.0
    public let frameSeconds: Double

    public init(frameSeconds: Double = 3200.0 / 24_000) { self.frameSeconds = frameSeconds }

    /// A threshold that follows the microphone: 15 dB above its noise floor (10th percentile of the
    /// frame levels), kept between -62 and -40 dBFS. A quiet headset mic (floor -79 dBFS, speech
    /// at -42) is missed by the fixed -40; a noisy room is not taken for speech.
    public static func adaptiveThreshold(_ levels: [Float]) -> Float {
        let s = levels.filter { $0 > -119 }.sorted()
        guard s.count >= 8 else { return -40 }
        return min(-40, max(-62, s[s.count / 10] + 15))
    }

    public static func level(_ x: ArraySlice<Float>) -> Float {
        guard !x.isEmpty else { return -120 }
        var sum: Float = 0
        for v in x { sum += v * v }
        return 10 * log10(max(sum / Float(x.count), 1e-12))
    }

    /// One level (dBFS) per frame of `frameSamples` samples.
    public static func levels(_ audio: [Float], frameSamples: Int) -> [Float] {
        stride(from: 0, to: audio.count, by: frameSamples).map { level(audio[$0 ..< min(audio.count, $0 + frameSamples)]) }
    }

    public func segments(_ levels: [Float]) -> [SpeechSegment] {
        var out: [SpeechSegment] = []
        var start: Int?
        var last = 0
        for (i, l) in levels.enumerated() {
            if l > thresholdDB {
                if start == nil { start = i }
                last = i
            } else if let s = start, i - last >= hangFrames {
                if last - s + 1 >= minFrames { out.append(segment(s, last)) }
                start = nil
            }
        }
        if let s = start, last - s + 1 >= minFrames { out.append(segment(s, last)) }
        return out
    }

    private func segment(_ first: Int, _ last: Int) -> SpeechSegment {
        SpeechSegment(start: Double(first) * frameSeconds, end: Double(last + 1) * frameSeconds)
    }

    /// Pairs each user utterance with the first Haru segment that starts after the utterance has begun,
    /// is still going when it ends, and starts before the next utterance. A segment answers at most one
    /// utterance: when the user talks over a sentence Haru had already started, that is an overlap, not
    /// a reply. `text` is (frame index, text) of Haru's frames.
    public func turns(user: [SpeechSegment], model: [SpeechSegment], text: [(Int, String)] = []) -> [Turn] {
        var used = Set<Int>()
        return user.enumerated().map { i, u in
            let next = i + 1 < user.count ? user[i + 1].start : .infinity
            let index = model.indices.first {
                !used.contains($0) && model[$0].start > u.start && model[$0].end > u.end && model[$0].start < next
            }
            if let index { used.insert(index) }
            let reply = index.map { model[$0] }
            var said = ""
            if let reply {
                for (index, t) in text {
                    let time = Double(index) * frameSeconds
                    if time >= reply.start - textLead, time < reply.end { said += t }
                }
            }
            return Turn(user: u, reply: reply, text: said.trimmingCharacters(in: .whitespacesAndNewlines))
        }
    }

    /// Turns of a conversation from both sides' audio at the same rate, aligned frame by frame.
    public func turns(userAudio: [Float], modelAudio: [Float], frameSamples: Int, text: [(Int, String)] = []) -> [Turn] {
        turns(user: segments(Self.levels(userAudio, frameSamples: frameSamples)),
              model: segments(Self.levels(modelAudio, frameSamples: frameSamples)), text: text)
    }
}

extension Array where Element == Turn {
    /// Median reply delay of the answered turns, and how many were answered.
    public var replySummary: (median: Double?, answered: Int) {
        let d = compactMap(\.delay).sorted()
        return (d.isEmpty ? nil : d[d.count / 2], d.count)
    }
}
