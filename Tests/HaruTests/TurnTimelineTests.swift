// TurnTimelineTests.swift - speech segments and reply delays from per-frame levels (no weights)
// Copyright 2026 Vincent Gourbin

import Testing
@testable import HaruCore

@Suite struct TurnTimelineTests {
    let d = TurnDetector()
    let T = 3200.0 / 24_000

    /// `pattern`: one character per frame, "#" speech, "." silence.
    func levels(_ pattern: String) -> [Float] { pattern.map { $0 == "#" ? -20 : -70 } }

    @Test func segmentsKeepShortPausesAndDropClicks() {
        // 3 frames, a 4-frame pause (< hang 5), 2 frames: one segment; then a 1-frame click: dropped.
        let s = d.segments(levels("..###....##........#......"))
        #expect(s.count == 1)
        #expect(abs(s[0].start - 2 * T) < 1e-9)
        #expect(abs(s[0].end - 11 * T) < 1e-9)
    }

    @Test func replyDelayAndText() {
        let user = d.segments(levels("####......................####........"))
        let haru = d.segments(levels(".......###............................"))
        let turns = d.turns(user: user, model: haru, text: [(6, " Hi"), (8, " there"), (30, " late")])
        #expect(turns.count == 2)
        #expect(abs(turns[0].delay! - 3 * T) < 1e-9)
        #expect(turns[0].text == "Hi there")
        #expect(turns[1].reply == nil)
        #expect(turns.replySummary.answered == 1)
    }

    @Test func overlapGivesNegativeDelay() {
        let user = d.segments(levels("######.........."))
        let haru = d.segments(levels("....######......"))
        let turns = d.turns(user: user, model: haru)
        #expect(abs(turns[0].delay! + 2 * T) < 1e-9)
    }

    @Test func talkingOverAnOngoingSentenceIsNotAnswered() {
        // Haru starts first; the user talks twice over it; then a real reply to the third utterance.
        let user = d.segments(levels("...###.......###............###.............."))
        let haru = d.segments(levels("##################...............#####......"))
        let turns = d.turns(user: user, model: haru)
        #expect(turns[0].reply == nil)
        #expect(turns[1].reply == nil)
        #expect(abs(turns[2].delay! - 2 * T) < 1e-9)
    }

    @Test func aSegmentAnswersOneUtterance() {
        // Two utterances in a row, one reply: it belongs to the first that it follows, not to both.
        let user = d.segments(levels("###........###.............."))
        let haru = d.segments(levels("..........########.........."))
        let turns = d.turns(user: user, model: haru)
        #expect(turns.compactMap(\.reply).count == 1)
    }

    @Test func levelOfSilenceAndFullScale() {
        #expect(TurnDetector.level([Float](repeating: 0, count: 3200)[...]) < -100)
        #expect(abs(TurnDetector.level([Float](repeating: 1, count: 3200)[...])) < 1e-4)
    }
}
