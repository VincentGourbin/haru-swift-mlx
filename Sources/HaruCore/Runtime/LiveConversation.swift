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
    /// Step times (seconds) of the frames produced so far, read after `stop()`.
    public private(set) var steps: [Double] = []

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
}

struct LivePump {
    let session: DuplexSession
    let conversation: LiveConversation
    let onFrame: @Sendable (Frame) -> Void
    var warm = 0.03

    /// Sleeps until `warm` before `date`, then polls.
    static func wait(until date: Date, warm: Double) {
        let sleep = date.timeIntervalSinceNow - warm
        if sleep > 0 { Thread.sleep(forTimeInterval: sleep) }
        while date.timeIntervalSinceNow > 0 { sched_yield() }
    }

    func run() {
        var raw: [Float] = []
        raw.reserveCapacity(48_000)
        var fed = 0  // 24 kHz samples fed
        let rate = Double(conversation.sampleRate)
        while conversation.drain(into: &raw) {
            let even = raw.count / 2 * 2
            if even > 0 {
                var pcm = [Float](repeating: 0, count: even / 2)
                for i in 0 ..< even / 2 { pcm[i] = (raw[2 * i] + raw[2 * i + 1]) * 0.5 }
                raw.removeFirst(even)
                session.feed(pcm)
                fed += pcm.count
            }
            while session.framesReady > 0 {
                guard let frame = try? session.step() else { return }
                conversation.record(frame.seconds)
                onFrame(frame)
                if frame.done { conversation.stop() }
            }
            // Time until the next user frame is complete: poll inside the warm window, else nap.
            let remaining = Double(conversation.frameSamples - fed % conversation.frameSamples) / rate
            if remaining > warm { Thread.sleep(forTimeInterval: 0.001) } else { sched_yield() }
        }
    }
}
