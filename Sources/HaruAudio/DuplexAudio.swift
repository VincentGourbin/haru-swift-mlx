// DuplexAudio.swift - microphone in, model audio out, reply latency on the audio clock
// Copied from higurashi-ai/Haru @ 7fe6b76, haru_mlx/swift/Shared/Audio.swift (Apache-2.0, © higurashi-ai);
// changes: split into files, `public` API for the in-process engine; logic unchanged.

import AudioToolbox
import CoreAudio
import Darwin
import Foundation

// MARK: - Input resampling (device rate -> 48 kHz)

/// Streaming linear interpolation; identity when the rates match.
struct Resampler {
    let ratio: Double  // input samples per output sample
    var position = 0.0  // next output position, in input samples from `last`
    var last: Float = 0

    init(from input: Double, to output: Double) { ratio = input / output }

    mutating func process(_ x: UnsafeBufferPointer<Float>, into out: inout [Float]) {
        if ratio == 1 {
            out.append(contentsOf: x)
            return
        }
        while position <= Double(x.count - 1) {  // samples indexed from -1 (`last`)
            let i = Int(position.rounded(.down))
            let frac = Float(position - Double(i))
            let a = i < 0 ? last : x[i]
            let b = x[i + 1]
            out.append(a + (b - a) * frac)
            position += ratio
        }
        position -= Double(x.count)
        if let tail = x.last { last = tail }
    }
}

// MARK: - Duplex audio

public struct LatencyReport: Sendable {
    public var frames = 0
    public var p50 = 0.0, p95 = 0.0, minimum = 0.0, maximum = 0.0
    public var underruns = 0
    public var caughtUpMs = 0.0
    public var parts: [(String, Double)] = []
}

public func median(_ v: [Double]) -> Double {
    let s = v.sorted()
    return s.isEmpty ? 0 : s[s.count / 2]
}

/// Microphone in and model audio out at 48 kHz, plus the reply latency on the
/// audio clock: from the last sample of each user frame at the ADC to the
/// first sample of the model frame answering it at the DAC.
public final class DuplexAudio: @unchecked Sendable {
    public let input: Device
    public let output: Device
    public let rate = 48_000.0
    let frameSamples: Int
    let bufferMs: Double
    public private(set) var lead: Double
    public private(set) var inputBufferMs = 0.0
    public private(set) var outputBufferMs = 0.0
    public private(set) var marginMs: Double
    public let player: Player
    public private(set) var inLatency: Double
    public private(set) var outLatency: Double
    /// One voice-processing unit for both directions (Apple's echo canceller, as FaceTime): the
    /// microphone signal has the speakers' output removed. Off: two raw HAL units (headphones).
    public let echoCancellation: Bool
    /// Rates the devices actually ran at once the units had started (Bluetooth headsets switch to
    /// a call rate when their microphone opens), and notes on what was reconfigured.
    public private(set) var runningInputRate = 0.0
    public private(set) var runningOutputRate = 0.0
    public private(set) var notes: [String] = []
    private let onAudio: (UnsafeBufferPointer<Float>) -> Void
    private var inUnit: AudioUnit?
    private var outUnit: AudioUnit?
    private let buffers: UnsafeMutableAudioBufferListPointer
    private let allocatedBuffers: Int
    private let capacity = 8192
    private var resampler: Resampler
    private var resampled: [Float] = []
    private var sent = 0  // 48 kHz samples sent so far

    // Measurements (audio and socket threads write, others read under the lock).
    private let statsLock = NSLock()
    private var adcEnd: [Int: Double] = [:]      // user frame k: ADC time of its last sample
    private var seen: [Int: Double] = [:]        // user frame k: when the input callback had it
    private var received: [Int: Double] = [:]    // model frame j: when it arrived from the server
    private var serverStep: [Int: Double] = [:]  // model frame j: step time the server sent
    public private(set) var inputLevel: Float = 0
    public private(set) var outputLevel: Float = 0
    /// Called on the audio threads with the 48 kHz input sent to the server
    /// (and the ADC time of its first sample), and with what is played (and the
    /// DAC time of its first sample). Set before start().
    public var inputTap: ((UnsafeBufferPointer<Float>, Double) -> Void)?
    public var outputTap: ((UnsafeBufferPointer<Float>, Double) -> Void)?

    public init(input: Device, output: Device, bufferMs: Double, marginMs: Double, frameSamples: Int,
         stepMs: Double, echoCancellation: Bool = false, onAudio: @escaping (UnsafeBufferPointer<Float>) -> Void) {
        self.echoCancellation = echoCancellation
        self.input = input
        self.output = output
        self.frameSamples = frameSamples
        self.bufferMs = bufferMs
        self.marginMs = marginMs
        self.stepMs = stepMs
        self.onAudio = onAudio
        player = Player(margin: Int(marginMs / 1000 * 48_000), rate: 48_000)
        inLatency = Double(deviceLatencyFrames(input.id, input: true)) / input.rate
        outLatency = Double(deviceLatencyFrames(output.id, input: false)) / output.rate
        // From a user frame's last sample at the ADC to the answering model frame
        // at the DAC: input buffer, step, ~5 ms for the socket and encoding the
        // frame's last chunk, output buffer and device, and the margin.
        lead = bufferMs / 1000 + stepMs / 1000 + 0.005 + bufferMs / 1000 + outLatency + marginMs / 1000
        allocatedBuffers = max(1, input.inputs)
        buffers = AudioBufferList.allocate(maximumBuffers: max(1, input.inputs))
        for i in 0..<max(1, input.inputs) {
            buffers[i] = AudioBuffer(mNumberChannels: 1, mDataByteSize: UInt32(capacity * 4),
                                     mData: UnsafeMutableRawPointer.allocate(byteCount: capacity * 4,
                                                                             alignment: 16))
        }
        resampler = Resampler(from: input.rate, to: 48_000)
        resampled.reserveCapacity(capacity * 2)
    }

    deinit {
        buffers.unsafeMutablePointer.pointee.mNumberBuffers = UInt32(allocatedBuffers)
        for b in buffers { b.mData?.deallocate() }
        free(buffers.unsafeMutablePointer)
    }

    private static func makeUnit(device: AudioDeviceID, input: Bool) throws -> AudioUnit {
        var desc = AudioComponentDescription(
            componentType: kAudioUnitType_Output, componentSubType: kAudioUnitSubType_HALOutput,
            componentManufacturer: kAudioUnitManufacturer_Apple, componentFlags: 0, componentFlagsMask: 0)
        var unit: AudioUnit?
        try ok(AudioComponentInstanceNew(AudioComponentFindNext(nil, &desc)!, &unit), "AudioComponentInstanceNew")
        let u = unit!
        var on: UInt32 = input ? 1 : 0, off: UInt32 = input ? 0 : 1
        try ok(AudioUnitSetProperty(u, kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Input, 1, &on, 4),
               "EnableIO input")
        try ok(AudioUnitSetProperty(u, kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Output, 0, &off, 4),
               "EnableIO output")
        var id = device
        try ok(AudioUnitSetProperty(u, kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global, 0,
                                    &id, UInt32(MemoryLayout<AudioDeviceID>.size)), "CurrentDevice")
        return u
    }

    private static func floatFormat(rate: Double, channels: Int) -> AudioStreamBasicDescription {
        AudioStreamBasicDescription(
            mSampleRate: rate, mFormatID: kAudioFormatLinearPCM,
            mFormatFlags: kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked | kAudioFormatFlagIsNonInterleaved,
            mBytesPerPacket: 4, mFramesPerPacket: 1, mBytesPerFrame: 4,
            mChannelsPerFrame: UInt32(channels), mBitsPerChannel: 32, mReserved: 0)
    }

    private let stepMs: Double

    public func start() throws {
        if echoCancellation { try startVoiceProcessing() } else { try startRaw() }
    }

    private func planLead() {
        // A frame can land just after an output cycle began: keep at least one
        // output cycle (plus 5 ms) of slack, and plan with the real buffers.
        marginMs = max(marginMs, outputBufferMs + 5)
        os_unfair_lock_lock(&player.lock)
        player.margin = Int(marginMs / 1000 * rate)
        os_unfair_lock_unlock(&player.lock)
        lead = inputBufferMs / 1000 + stepMs / 1000 + 0.005 + outputBufferMs / 1000 + outLatency
            + marginMs / 1000
    }

    private var inputCallback: AURenderCallbackStruct {
        AURenderCallbackStruct(inputProc: { refCon, flags, ts, bus, frames, _ in
            Unmanaged<DuplexAudio>.fromOpaque(refCon).takeUnretainedValue()
                .handleInput(flags, ts, bus, frames)
        }, inputProcRefCon: Unmanaged.passUnretained(self).toOpaque())
    }

    private var renderCallback: AURenderCallbackStruct {
        AURenderCallbackStruct(inputProc: { refCon, _, ts, _, frames, data in
            Unmanaged<DuplexAudio>.fromOpaque(refCon).takeUnretainedValue()
                .handleOutput(ts, frames, data!)
        }, inputProcRefCon: Unmanaged.passUnretained(self).toOpaque())
    }

    /// Two HAL units, raw signal. The input side cannot convert rates, so the client format follows
    /// the device's rate as it is once the unit runs (a Bluetooth headset switches when its mic opens).
    private func startRaw() throws {
        let size = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
        let inU = try Self.makeUnit(device: input.id, input: true)
        var inCallback = inputCallback
        try ok(AudioUnitSetProperty(inU, kAudioOutputUnitProperty_SetInputCallback, kAudioUnitScope_Global, 0,
                                    &inCallback, UInt32(MemoryLayout<AURenderCallbackStruct>.size)),
               "input callback")
        func configureInput(rate deviceRate: Double) throws {
            var inFormat = Self.floatFormat(rate: deviceRate, channels: max(1, input.inputs))
            try ok(AudioUnitSetProperty(inU, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Output, 1,
                                        &inFormat, size), "input format")
            resampler = Resampler(from: deviceRate, to: 48_000)
            runningInputRate = deviceRate
        }
        try configureInput(rate: currentRate(input.id))
        try ok(AudioUnitInitialize(inU), "initialize input")
        inUnit = inU  // the callback renders from it
        try ok(AudioOutputUnitStart(inU), "start input")
        // Opening the microphone may have changed the device's rate: reconfigure once if so.
        for _ in 0 ..< 20 {
            let now = currentRate(input.id)
            if now != runningInputRate {
                notes.append(String(format: "input rate changed to %.0f Hz when the microphone opened (was %.0f)", now, runningInputRate))
                AudioOutputUnitStop(inU)
                AudioUnitUninitialize(inU)
                try configureInput(rate: now)
                try ok(AudioUnitInitialize(inU), "initialize input")
                try ok(AudioOutputUnitStart(inU), "start input")
                break
            }
            Thread.sleep(forTimeInterval: 0.01)
        }

        let outU = try Self.makeUnit(device: output.id, input: false)
        var outFormat = Self.floatFormat(rate: rate, channels: 1)  // the unit converts to the device
        try ok(AudioUnitSetProperty(outU, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input, 0,
                                    &outFormat, size), "output format")
        var outCallback = renderCallback
        try ok(AudioUnitSetProperty(outU, kAudioUnitProperty_SetRenderCallback, kAudioUnitScope_Input, 0,
                                    &outCallback, UInt32(MemoryLayout<AURenderCallbackStruct>.size)),
               "render callback")
        try ok(AudioUnitInitialize(outU), "initialize output")
        outUnit = outU
        measureDevices()
        try ok(AudioOutputUnitStart(outU), "start output")
    }

    /// Buffers, rates and latencies as the devices run now (after any switch).
    private func measureDevices() {
        runningOutputRate = currentRate(output.id)
        if runningInputRate == 0 { runningInputRate = currentRate(input.id) }
        let inFrames = setBufferFrames(input.id, UInt32(runningInputRate * bufferMs / 1000))
        let outFrames = setBufferFrames(output.id, UInt32(runningOutputRate * bufferMs / 1000))
        inputBufferMs = Double(inFrames) / runningInputRate * 1000
        outputBufferMs = Double(outFrames) / runningOutputRate * 1000
        inLatency = Double(deviceLatencyFrames(input.id, input: true)) / runningInputRate
        outLatency = Double(deviceLatencyFrames(output.id, input: false)) / runningOutputRate
        planLead()
    }

    /// Apple's voice-processing unit: one unit, microphone on bus 1, playback on bus 0, so the echo
    /// canceller sees what is played. Automatic gain is turned off (it would change what the model
    /// hears from one sentence to the next); other audio is ducked as little as the system allows.
    private func startVoiceProcessing() throws {
        var desc = AudioComponentDescription(
            componentType: kAudioUnitType_Output, componentSubType: kAudioUnitSubType_VoiceProcessingIO,
            componentManufacturer: kAudioUnitManufacturer_Apple, componentFlags: 0, componentFlagsMask: 0)
        var unit: AudioUnit?
        try ok(AudioComponentInstanceNew(AudioComponentFindNext(nil, &desc)!, &unit), "voice processing unit")
        let u = unit!
        let size = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
        var on: UInt32 = 1
        try ok(AudioUnitSetProperty(u, kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Input, 1, &on, 4), "EnableIO input")
        try ok(AudioUnitSetProperty(u, kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Output, 0, &on, 4), "EnableIO output")
        // Devices: element 1 is the microphone side, element 0 the speaker side.
        var inID = input.id, outID = output.id
        let idSize = UInt32(MemoryLayout<AudioDeviceID>.size)
        let inStatus = AudioUnitSetProperty(u, kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global, 1, &inID, idSize)
        let outStatus = AudioUnitSetProperty(u, kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global, 0, &outID, idSize)
        if inStatus != noErr || outStatus != noErr {
            notes.append("voice processing: device selection refused (\(inStatus), \(outStatus)), system defaults used")
        }
        var agc: UInt32 = 0
        AudioUnitSetProperty(u, kAUVoiceIOProperty_VoiceProcessingEnableAGC, kAudioUnitScope_Global, 1, &agc, 4)
        if #available(macOS 14.0, *) {
            var ducking = AUVoiceIOOtherAudioDuckingConfiguration(mEnableAdvancedDucking: true,
                                                                  mDuckingLevel: .min)
            AudioUnitSetProperty(u, kAUVoiceIOProperty_OtherAudioDuckingConfiguration, kAudioUnitScope_Global, 0,
                                 &ducking, UInt32(MemoryLayout<AUVoiceIOOtherAudioDuckingConfiguration>.size))
        }
        // The unit converts both directions to and from 48 kHz mono float.
        var format = Self.floatFormat(rate: rate, channels: 1)
        try ok(AudioUnitSetProperty(u, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Output, 1, &format, size), "mic format")
        try ok(AudioUnitSetProperty(u, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input, 0, &format, size), "speaker format")
        var inCallback = inputCallback, outCallback = renderCallback
        try ok(AudioUnitSetProperty(u, kAudioOutputUnitProperty_SetInputCallback, kAudioUnitScope_Global, 1,
                                    &inCallback, UInt32(MemoryLayout<AURenderCallbackStruct>.size)), "input callback")
        try ok(AudioUnitSetProperty(u, kAudioUnitProperty_SetRenderCallback, kAudioUnitScope_Input, 0,
                                    &outCallback, UInt32(MemoryLayout<AURenderCallbackStruct>.size)), "render callback")
        buffers.unsafeMutablePointer.pointee.mNumberBuffers = 1  // mono after processing
        try ok(AudioUnitInitialize(u), "initialize voice processing")
        resampler = Resampler(from: rate, to: 48_000)
        runningInputRate = currentRate(input.id)
        inUnit = u
        measureDevices()
        try ok(AudioOutputUnitStart(u), "start voice processing")
    }

    public func stop() {
        if let u = outUnit { AudioOutputUnitStop(u); AudioComponentInstanceDispose(u) }
        if let u = inUnit { AudioOutputUnitStop(u); AudioComponentInstanceDispose(u) }  // the one unit with echo cancellation
        outUnit = nil
        inUnit = nil
    }

    private func handleInput(_ flags: UnsafeMutablePointer<AudioUnitRenderActionFlags>,
                             _ ts: UnsafePointer<AudioTimeStamp>, _ bus: UInt32, _ frames: UInt32) -> OSStatus {
        for i in 0..<buffers.count { buffers[i].mDataByteSize = frames * 4 }
        let status = AudioUnitRender(inUnit!, flags, ts, bus, frames, buffers.unsafeMutablePointer)
        if status != noErr { return status }
        let seenAt = now()
        let adc = hostSeconds(ts.pointee.mHostTime) - inLatency  // ADC time of the first sample
        if sent == 0 {
            os_unfair_lock_lock(&player.lock)
            player.startAt = adc + lead
            os_unfair_lock_unlock(&player.lock)
        }
        let mono = UnsafeBufferPointer(start: buffers[0].mData!.assumingMemoryBound(to: Float.self),
                                       count: Int(frames))
        inputLevel = max(rms(mono), inputLevel * levelDecay(frames))
        resampled.removeAll(keepingCapacity: true)
        resampler.process(mono, into: &resampled)
        let start = sent
        sent += resampled.count
        var k = start / frameSamples
        while (k + 1) * frameSamples <= sent {
            let last = (k + 1) * frameSamples - 1 - start
            statsLock.lock()
            adcEnd[k] = adc + Double(last) / rate
            seen[k] = seenAt
            statsLock.unlock()
            k += 1
        }
        resampled.withUnsafeBufferPointer {
            onAudio($0)
            inputTap?($0, adc)
        }
        return noErr
    }

    private func handleOutput(_ ts: UnsafePointer<AudioTimeStamp>, _ frames: UInt32,
                              _ data: UnsafeMutablePointer<AudioBufferList>) -> OSStatus {
        let list = UnsafeMutableAudioBufferListPointer(data)
        let out = UnsafeMutableBufferPointer(start: list[0].mData!.assumingMemoryBound(to: Float.self),
                                             count: Int(frames))
        let dac = hostSeconds(ts.pointee.mHostTime) + outLatency
        player.pull(out, dac: dac)
        outputTap?(UnsafeBufferPointer(out), dac)
        outputLevel = max(rms(UnsafeBufferPointer(out)), outputLevel * levelDecay(frames))
        return noErr
    }

    /// Peak-hold with a ~80 ms release, so readers sampling at screen rate see
    /// speech rather than whichever 2 ms block came last.
    private func levelDecay(_ frames: UInt32) -> Float {
        Float(exp(-Double(frames) / rate / 0.08))
    }

    private func rms(_ x: UnsafeBufferPointer<Float>) -> Float {
        guard !x.isEmpty else { return 0 }
        var sum: Float = 0
        for v in x { sum += v * v }
        return (sum / Float(x.count)).squareRoot()
    }

    /// A model frame from the server (`step` is the server's step time).
    public func receive(index: Int, pcm: [Float], step: Double) {
        statsLock.lock()
        received[index] = now()
        serverStep[index] = step
        statsLock.unlock()
        player.push(index, pcm)
    }

    private func latencies() -> [Int] {
        os_unfair_lock_lock(&player.lock)
        let dac = player.dacStart, written = player.writtenAt
        os_unfair_lock_unlock(&player.lock)
        statsLock.lock()
        defer { statsLock.unlock() }
        return dac.keys.sorted().filter { j in
            j >= 1 && adcEnd[j - 1] != nil && received[j] != nil && written[j] != nil
        }
    }

    /// Median reply latency over the last frames played, in seconds. Copies only
    /// the player's short `recent` list, so the audio thread is not held up.
    public func recentLatency() -> Double? {
        os_unfair_lock_lock(&player.lock)
        let recent = player.recent
        os_unfair_lock_unlock(&player.lock)
        statsLock.lock()
        let values = recent.compactMap { j, dac in j >= 1 ? adcEnd[j - 1].map { dac - $0 } : nil }
        statsLock.unlock()
        return values.isEmpty ? nil : median(values)
    }

    public func report() -> LatencyReport {
        let frames = latencies()
        os_unfair_lock_lock(&player.lock)
        let dac = player.dacStart, written = player.writtenAt
        var r = LatencyReport()
        r.underruns = player.underruns
        r.caughtUpMs = Double(player.compressed) / rate * 1000
        os_unfair_lock_unlock(&player.lock)
        statsLock.lock()
        defer { statsLock.unlock() }
        let total = frames.map { dac[$0]! - adcEnd[$0 - 1]! }.sorted()
        guard !total.isEmpty else { return r }
        r.frames = total.count
        r.p50 = total[total.count / 2]
        r.p95 = total[min(total.count - 1, Int(Double(total.count) * 0.95))]
        r.minimum = total[0]
        r.maximum = total[total.count - 1]
        r.parts = [
            ("input (ADC -> callback)", median(frames.map { seen[$0 - 1]! - adcEnd[$0 - 1]! })),
            ("server (callback -> frame received)", median(frames.map { received[$0]! - seen[$0 - 1]! })),
            ("  of which step (server)", median(frames.map { serverStep[$0]! })),
            ("output queue (lead/margin)", median(frames.map { written[$0]! - received[$0]! })),
            ("output (render -> DAC)", median(frames.map { dac[$0]! - written[$0]! })),
        ]
        return r
    }
}
