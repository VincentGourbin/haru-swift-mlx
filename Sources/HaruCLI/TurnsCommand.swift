// TurnsCommand.swift - `haru turns`: who spoke when in a stereo conversation, and each reply's delay
// Copyright 2026 Vincent Gourbin

import ArgumentParser
import AVFoundation
import Foundation
import HaruCore

struct Turns: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Turns of a stereo conversation (user left, Haru right, 24 kHz): segments and reply delays.")

    @Argument(help: "Stereo WAV written by `haru run --stereo`, `haru simulate` or the app.") var conversation: String
    @Option(help: "Per-frame JSON lines (`--transcript` of `haru run`) to show what Haru said.") var transcript: String?
    @Option(help: "Speech threshold in dBFS.") var thresholdDb: Float = -40
    @Flag(help: "One JSON object instead of a table.") var json = false

    func run() throws {
        let file = try AVAudioFile(forReading: URL(fileURLWithPath: (conversation as NSString).expandingTildeInPath),
                                   commonFormat: .pcmFormatFloat32, interleaved: false)
        guard file.processingFormat.channelCount == 2, file.processingFormat.sampleRate == 24_000 else {
            throw ValidationError("expected a stereo 24 kHz WAV")
        }
        let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length))!
        try file.read(into: buffer)
        let n = Int(buffer.frameLength)
        let user = Array(UnsafeBufferPointer(start: buffer.floatChannelData![0], count: n))
        let model = Array(UnsafeBufferPointer(start: buffer.floatChannelData![1], count: n))
        var text: [(Int, String)] = []
        if let transcript {
            for line in try String(contentsOfFile: (transcript as NSString).expandingTildeInPath, encoding: .utf8).split(separator: "\n") {
                if let o = try JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any],
                   let f = o["frame"] as? Int, let t = o["text"] as? String, !t.isEmpty { text.append((f, t)) }
            }
        }
        var detector = TurnDetector()
        detector.thresholdDB = thresholdDb
        let turns = detector.turns(userAudio: user, modelAudio: model, frameSamples: 3200, text: text)
        let summary = turns.replySummary
        if json {
            let data = try JSONEncoder().encode(turns)
            let list = try JSONSerialization.jsonObject(with: data)
            let out: [String: Any] = ["turns": list, "answered": summary.answered, "median_delay": summary.median ?? NSNull()]
            print(String(decoding: try JSONSerialization.data(withJSONObject: out, options: [.sortedKeys]), as: UTF8.self))
            return
        }
        for (i, t) in turns.enumerated() {
            let you = String(format: "%2d  you %6.2f-%6.2f s", i + 1, t.user.start, t.user.end)
            if let r = t.reply, let d = t.delay {
                print(you + String(format: "   haru %6.2f-%6.2f s  delay %+5.2f s  %@", r.start, r.end, d, t.text))
            } else {
                print(you + "   (no reply before the next utterance)")
            }
        }
        print(String(format: "%d/%d answered, median delay %@", summary.answered, turns.count,
                     summary.median.map { String(format: "%+.2f s", $0) } ?? "—"))
    }
}
