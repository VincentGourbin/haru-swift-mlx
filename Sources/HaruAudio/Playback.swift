// Playback.swift - the playback schedule and WSOLA catch-up
// Copied from higurashi-ai/Haru @ 7fe6b76, haru_mlx/swift/Shared/Audio.swift (Apache-2.0, © higurashi-ai);
// changes: split into files, `public` API for the in-process engine; logic unchanged.

import AudioToolbox
import CoreAudio
import Darwin
import Foundation

// MARK: - Catch-up

/// Shorten `x` by `rate` (0..0.25) without changing its pitch (WSOLA): Hann
/// windows at 50 % overlap, each taken near its nominal (faster) input position
/// where it best continues the previous one. The first half window and the
/// last half window keep the input's own samples, so frames still join.
public func wsolaCompress(_ x: [Float], rate: Double, win: Int = 960, tol: Int = 240) -> [Float] {
    let n = x.count
    let outLen = Int((Double(n) * (1 - rate)).rounded())
    let hop = win / 2
    if rate <= 0 || n < 4 * win || outLen <= win { return x }
    let speed = Double(n) / Double(outLen)
    let w = (0..<win).map { Float(0.5 - 0.5 * cos(2 * Double.pi * Double($0) / Double(win))) }
    var out = [Float](repeating: 0, count: outLen + win)
    var wsum = [Float](repeating: 0, count: outLen + win)
    for i in 0..<win {
        let wi = i < hop ? 1 : w[i]
        out[i] += x[i] * wi
        wsum[i] += wi
    }
    var prev = 0
    var k = 1
    while k * hop + hop <= outLen {
        let nominal = Int((Double(k * hop) * speed).rounded())
        let natural = prev + hop  // what would follow the previous window unchanged
        let hi = min(n - win, nominal + tol)
        let lo = min(max(0, nominal - tol), hi)
        var best = (lo...hi).contains(nominal) ? nominal : lo
        if natural + win <= n {
            var bestScore = -Float.infinity
            for start in lo...hi {
                var score: Float = 0
                for i in stride(from: 0, to: win, by: 4) { score += x[start + i] * x[natural + i] }
                if score > bestScore {
                    bestScore = score
                    best = start
                }
            }
        }
        for i in 0..<win {
            out[k * hop + i] += x[best + i] * w[i]
            wsum[k * hop + i] += w[i]
        }
        prev = best
        k += 1
    }
    var y = (0..<outLen).map { wsum[$0] > 1e-3 ? out[$0] / wsum[$0] : 0 }
    for i in 0..<hop {  // crossfade into the input's tail
        let r = Float(i + 1) / Float(hop)
        y[outLen - hop + i] = y[outLen - hop + i] * (1 - r) + x[n - hop + i] * r
    }
    return y
}

// MARK: - Playback

/// Model frame 0 starts playing at `startAt` (the first input sample's ADC
/// time plus the time model frame 1 needs), so frame 1 is ready when frame 0
/// ends whatever order the devices started in. After an underrun, `margin` of
/// silence restores the slack. Catch-up: while more than `margin` + 5 ms is
/// queued when a frame arrives (after an underrun, or frames that came late
/// together), the frame is shortened by up to 10 % (20 % above 150 ms) with
/// WSOLA until playback is back on schedule.
public final class Player: @unchecked Sendable {
    var lock = os_unfair_lock()
    var chunks: [(Int, [Float])] = []
    var offset = 0
    var pad = 0
    var margin: Int
    let rate: Double
    var startAt: Double? = nil  // DAC time of model frame 0
    var started = false
    var dry = false
    var underruns = 0
    var compressed = 0  // samples removed by catch-up
    var dacStart: [Int: Double] = [:]
    var writtenAt: [Int: Double] = [:]
    var recent: [(Int, Double)] = []  // last frames' (index, DAC start), for live display

    init(margin: Int, rate: Double) {
        self.margin = margin
        self.rate = rate
    }

    func push(_ index: Int, _ pcm: [Float]) {
        os_unfair_lock_lock(&lock)
        let queued = chunks.reduce(0) { $0 + $1.1.count } - offset + pad
        let playing = started && !dacStart.isEmpty
        os_unfair_lock_unlock(&lock)
        var frame = pcm
        let excess = queued - margin
        if playing && excess > Int(0.005 * rate) {
            let cap = excess > Int(0.150 * rate) ? 0.20 : 0.10
            frame = wsolaCompress(pcm, rate: min(cap, Double(excess) / Double(pcm.count)))
        }
        os_unfair_lock_lock(&lock)
        compressed += pcm.count - frame.count
        chunks.append((index, frame))
        os_unfair_lock_unlock(&lock)
    }

    func pull(_ out: UnsafeMutableBufferPointer<Float>, dac: Double) {
        let t = now()
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }
        if !started {
            let wait = startAt.map { ($0 - dac) * rate } ?? .infinity
            if wait >= Double(out.count) {
                for i in 0..<out.count { out[i] = 0 }
                return
            }
            started = true
            pad = max(0, Int(wait))
        }
        if dry && !chunks.isEmpty {
            dry = false
            pad = margin
        }
        var written = 0
        while written < out.count {
            if pad > 0 {
                let take = min(pad, out.count - written)
                for i in 0..<take { out[written + i] = 0 }
                written += take
                pad -= take
                continue
            }
            guard !chunks.isEmpty else { break }
            let (index, pcm) = chunks[0]
            if offset == 0 {
                dacStart[index] = dac + Double(written) / rate
                writtenAt[index] = t
                recent.append((index, dacStart[index]!))
                if recent.count > 64 { recent.removeFirst() }
            }
            let take = min(pcm.count - offset, out.count - written)
            for i in 0..<take { out[written + i] = pcm[offset + i] }
            written += take
            offset += take
            if offset == pcm.count {
                chunks.removeFirst()
                offset = 0
            }
        }
        if written < out.count {
            for i in written..<out.count { out[i] = 0 }
            if !dry {
                dry = true
                underruns += 1
            }
        }
    }
}

