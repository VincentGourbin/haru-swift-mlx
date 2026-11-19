// LiveConversation.swift - microphone samples in (any thread), frames out (MLX thread)
// Copyright 2026 Vincent Gourbin
//
// The audio thread only copies 48 kHz samples into a preallocated buffer under a short lock (no
// allocation, no MLX: pitfall 22). The pump, on the MLX thread, drains it, halves the rate by pairwise
// means (upstream `downsample_2x`), feeds the session as audio arrives (the encoder front runs on every
// whole hop) and steps as soon as a user frame is complete. Before a frame is due it polls instead of
// sleeping, so the core is awake when the step starts (upstream worker.py `warm`).

import Foundation
import os

public final class LiveConversation: @unchecked Sendable {
    public let sampleRate: Int
    public let frameSamples: Int
    private var lock = os_unfair_lock()
    private var buffer: [Float]
    private var count = 0
    private var dropped = 0
    fileprivate var stopped = false
    private let finished = DispatchSemaphore(value: 0)
    /// Step times (seconds) of the frames produced so far, read after `stop()`.
    public private(set) var steps: [Double] = []
    /// When true, the 24 kHz user audio fed to the model is kept (`userAudio`, read after `stop()`).
    public var recordsUser = false
    public private(set) var userAudio: [Float] = []

    init(sampleRate: Int, frameSamples: Int) {
        self.sampleRate = sampleRate
        self.frameSamples = frameSamples
        buffer = [Float](repeating: 0, count: 48_000 * 2)  // two seconds of slack at 48 kHz
    }

    /// 48 kHz mono samples from the microphone (audio thread).
    public func push48k(_ samples: UnsafeBufferPointer<Float>) {
        os_unfair_lock_lock(&lock)
        let n = min(samples.count, buffer.count - count)
        if n > 0 {
            buffer.withUnsafeMutableBufferPointer { dst in
                (dst.baseAddress! + count).update(from: samples.baseAddress!, count: n)
            }
            count += n
        }
        dropped += samples.count - n
        os_unfair_lock_unlock(&lock)
    }

    public func stop() {
        os_unfair_lock_lock(&lock)
        stopped = true
        os_unfair_lock_unlock(&lock)
    }

    /// Stops and waits (at most `timeout`) for the pump to leave the MLX thread: `steps` and
    /// `userAudio` are then complete and no longer written.
    @discardableResult
    public func stopAndWait(timeout: Double = 2) -> Bool {
        stop()
        return finished.wait(timeout: .now() + timeout) == .success
    }

    fileprivate func pumpFinished() { finished.signal() }

    /// Samples lost because the pump fell more than two seconds behind.
    public var droppedSamples: Int {
        os_unfair_lock_lock(&lock); defer { os_unfair_lock_unlock(&lock) }
        return dropped
    }

    fileprivate func drain(into out: inout [Float]) -> Bool {
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }
        if count > 0 {
            buffer.withUnsafeBufferPointer { out.append(contentsOf: UnsafeBufferPointer(start: $0.baseAddress!, count: count)) }
            count = 0
        }
        return !stopped
    }

    fileprivate func record(_ seconds: Double) { steps.append(seconds) }
    fileprivate func keep(_ pcm: [Float]) { if recordsUser { userAudio.append(contentsOf: pcm) } }
}

struct LivePump {
    let session: DuplexSession
    let conversation: LiveConversation
    let onFrame: @Sendable (Frame) -> Void
    /// Level (dBFS) of each complete 24 kHz user frame, with its index, as it is fed (MLX thread).
    var onUserLevel: (@Sendable (Int, Float) -> Void)?
    var prompter: Prompter?
    /// Seconds before a user frame completes during which the pump polls instead of napping. Default:
    /// always (more than a frame). Napping 1 ms between audio blocks let the CPU clock drop after ~20 s
    /// and the step rose from 46 to 57 ms (measured 2026-11-17; docs/knowledge/pitfalls/cpu-clock-drops-
    /// between-frames.md); polling keeps it at 44 ms for one busy core. HARU_WARM_MS overrides it.
    var warm = (ProcessInfo.processInfo.environment["HARU_WARM_MS"].flatMap(Double.init) ?? 140) / 1000

    /// Sleeps until `warm` before `date`, then polls.
    static func wait(until date: Date, warm: Double) {
        let sleep = date.timeIntervalSinceNow - warm
        if sleep > 0 { Thread.sleep(forTimeInterval: sleep) }
        while date.timeIntervalSinceNow > 0 { sched_yield() }
    }

    func run() {
        defer { conversation.pumpFinished() }
        var raw: [Float] = []
        raw.reserveCapacity(48_000)
        var fed = 0  // 24 kHz samples fed
        var energy: Float = 0  // sum of squares of the user frame being filled
        var userLevel: Float = -120  // level of the last complete user frame (the one the next step consumes)
        let rate = Double(conversation.sampleRate)
        while conversation.drain(into: &raw) {
            // One pool per iteration: the pump is a single job for the whole conversation, so the Metal
            // objects MLX autoreleases (~50 small evals per frame) are released as it goes.
            let keepGoing: Bool = autoreleasepool {
            let even = raw.count / 2 * 2
            if even > 0 {
                var pcm = [Float](repeating: 0, count: even / 2)
                for i in 0 ..< even / 2 { pcm[i] = (raw[2 * i] + raw[2 * i + 1]) * 0.5 }
                raw.removeFirst(even)
                session.feed(pcm)
                conversation.keep(pcm)
                if onUserLevel != nil || prompter != nil {
                    let size = conversation.frameSamples
                    for (i, v) in pcm.enumerated() {
                        energy += v * v
                        if (fed + i + 1) % size == 0 {
                            userLevel = 10 * log10(max(energy / Float(size), 1e-12))
                            onUserLevel?((fed + i + 1) / size - 1, userLevel)
                            energy = 0
                        }
                    }
                }
                fed += pcm.count
            }
            while session.framesReady > 0 {
                guard let frame = try? session.step() else { return false }
                conversation.record(frame.seconds)
                prompter?.observe(index: frame.index, output: frame.audio, userLevel: userLevel)
                onFrame(frame)
                if frame.done { conversation.stop() }
            }
            return true
            }
            guard keepGoing else { return }
            // Time until the next user frame is complete: poll inside the warm window, else nap.
            let remaining = Double(conversation.frameSamples - fed % conversation.frameSamples) / rate
            if remaining > warm { Thread.sleep(forTimeInterval: 0.001) } else { sched_yield() }
        }
    }
}
