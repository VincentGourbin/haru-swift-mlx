// Devices.swift - Core Audio devices and the host clock
// Copied from higurashi-ai/Haru @ 7fe6b76, haru_mlx/swift/Shared/Audio.swift (Apache-2.0, © higurashi-ai);
// changes: split into files, `public` API for the in-process engine; logic unchanged.

import AudioToolbox
import CoreAudio
import Darwin
import Foundation

// MARK: - Clock

let timebase: mach_timebase_info_data_t = {
    var info = mach_timebase_info_data_t()
    mach_timebase_info(&info)
    return info
}()

public func hostSeconds(_ host: UInt64) -> Double {
    Double(host) * Double(timebase.numer) / Double(timebase.denom) / 1e9
}

public func now() -> Double { hostSeconds(mach_absolute_time()) }

// MARK: - Devices

public struct AudioError: Error, CustomStringConvertible {
    public let description: String
}

func ok(_ status: OSStatus, _ what: String) throws {
    if status != noErr { throw AudioError(description: "\(what) failed (\(status))") }
}

func property<T>(_ id: AudioObjectID, _ selector: AudioObjectPropertySelector,
                 _ scope: AudioObjectPropertyScope, _ value: T) -> T {
    var address = AudioObjectPropertyAddress(
        mSelector: selector, mScope: scope, mElement: kAudioObjectPropertyElementMain)
    var result = value
    var size = UInt32(MemoryLayout<T>.size)
    AudioObjectGetPropertyData(id, &address, 0, nil, &size, &result)
    return result
}

public struct Device: Hashable, Sendable {
    public let id: AudioDeviceID
    public let name: String
    public let inputs: Int
    public let outputs: Int
    public let rate: Double
}

private func channels(_ id: AudioDeviceID, _ scope: AudioObjectPropertyScope) -> Int {
    var address = AudioObjectPropertyAddress(
        mSelector: kAudioDevicePropertyStreamConfiguration, mScope: scope,
        mElement: kAudioObjectPropertyElementMain)
    var size: UInt32 = 0
    AudioObjectGetPropertyDataSize(id, &address, 0, nil, &size)
    let list = UnsafeMutableRawPointer.allocate(byteCount: Int(max(size, 16)), alignment: 16)
    defer { list.deallocate() }
    AudioObjectGetPropertyData(id, &address, 0, nil, &size, list)
    let buffers = UnsafeMutableAudioBufferListPointer(
        list.assumingMemoryBound(to: AudioBufferList.self))
    return buffers.reduce(0) { $0 + Int($1.mNumberChannels) }
}

private func deviceName(_ id: AudioDeviceID) -> String {
    var address = AudioObjectPropertyAddress(
        mSelector: kAudioObjectPropertyName, mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain)
    var name: Unmanaged<CFString>?
    var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
    AudioObjectGetPropertyData(id, &address, 0, nil, &size, &name)
    return name?.takeRetainedValue() as String? ?? "?"
}

public func audioDevices() -> [Device] {
    var address = AudioObjectPropertyAddress(
        mSelector: kAudioHardwarePropertyDevices, mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain)
    let system = AudioObjectID(kAudioObjectSystemObject)
    var size: UInt32 = 0
    AudioObjectGetPropertyDataSize(system, &address, 0, nil, &size)
    var ids = [AudioDeviceID](repeating: 0, count: Int(size) / MemoryLayout<AudioDeviceID>.size)
    AudioObjectGetPropertyData(system, &address, 0, nil, &size, &ids)
    return ids.map { id in
        Device(id: id, name: deviceName(id),
               inputs: channels(id, kAudioObjectPropertyScopeInput),
               outputs: channels(id, kAudioObjectPropertyScopeOutput),
               rate: property(id, kAudioDevicePropertyNominalSampleRate,
                              kAudioObjectPropertyScopeGlobal, Float64(0)))
    }
}

/// The device's current nominal sample rate (it can change once its microphone is in use: Bluetooth).
public func currentRate(_ id: AudioDeviceID) -> Double {
    property(id, kAudioDevicePropertyNominalSampleRate, kAudioObjectPropertyScopeGlobal, Float64(0))
}

public func defaultDevice(input: Bool) -> AudioDeviceID {
    property(AudioObjectID(kAudioObjectSystemObject),
             input ? kAudioHardwarePropertyDefaultInputDevice : kAudioHardwarePropertyDefaultOutputDevice,
             kAudioObjectPropertyScopeGlobal, AudioDeviceID(0))
}

/// The named device (or the system default) that has inputs / outputs.
public func findDevice(_ name: String?, input: Bool) -> Device? {
    let id = defaultDevice(input: input)
    return audioDevices().first { d in
        (name == nil || name!.isEmpty ? d.id == id : d.name == name)
            && (input ? d.inputs > 0 : d.outputs > 0)
    }
}

/// Device latency + safety offset + stream latency, in frames, as the driver states them.
func deviceLatencyFrames(_ id: AudioDeviceID, input: Bool) -> UInt32 {
    let scope = input ? kAudioObjectPropertyScopeInput : kAudioObjectPropertyScopeOutput
    let latency: UInt32 = property(id, kAudioDevicePropertyLatency, scope, UInt32(0))
    let safety: UInt32 = property(id, kAudioDevicePropertySafetyOffset, scope, UInt32(0))
    var address = AudioObjectPropertyAddress(
        mSelector: kAudioDevicePropertyStreams, mScope: scope,
        mElement: kAudioObjectPropertyElementMain)
    var size: UInt32 = 0
    AudioObjectGetPropertyDataSize(id, &address, 0, nil, &size)
    var streams = [AudioStreamID](repeating: 0, count: Int(size) / MemoryLayout<AudioStreamID>.size)
    AudioObjectGetPropertyData(id, &address, 0, nil, &size, &streams)
    let stream: UInt32 = streams.first.map {
        property($0, kAudioStreamPropertyLatency, kAudioObjectPropertyScopeGlobal, UInt32(0))
    } ?? 0
    return latency + safety + stream
}

/// Ask for `frames` per IO cycle and return what the device actually uses
/// (Bluetooth devices, for one, keep larger buffers).
func setBufferFrames(_ id: AudioDeviceID, _ frames: UInt32) -> UInt32 {
    var address = AudioObjectPropertyAddress(
        mSelector: kAudioDevicePropertyBufferFrameSize, mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain)
    var value = frames
    AudioObjectSetPropertyData(id, &address, 0, nil, UInt32(MemoryLayout<UInt32>.size), &value)
    return property(id, kAudioDevicePropertyBufferFrameSize, kAudioObjectPropertyScopeGlobal, frames)
}


/// Calls `onChange` on the main queue when a device appears or goes, or a system default changes.
public final class DeviceWatcher: @unchecked Sendable {
    private let selectors: [AudioObjectPropertySelector] = [
        kAudioHardwarePropertyDevices, kAudioHardwarePropertyDefaultInputDevice, kAudioHardwarePropertyDefaultOutputDevice,
    ]
    private var blocks: [(AudioObjectPropertyAddress, AudioObjectPropertyListenerBlock)] = []

    public init(onChange: @escaping @Sendable () -> Void) {
        for selector in selectors {
            var address = AudioObjectPropertyAddress(mSelector: selector, mScope: kAudioObjectPropertyScopeGlobal,
                                                     mElement: kAudioObjectPropertyElementMain)
            let block: AudioObjectPropertyListenerBlock = { _, _ in onChange() }
            AudioObjectAddPropertyListenerBlock(AudioObjectID(kAudioObjectSystemObject), &address, .main, block)
            blocks.append((address, block))
        }
    }

    deinit {
        for (address, block) in blocks {
            var a = address
            AudioObjectRemovePropertyListenerBlock(AudioObjectID(kAudioObjectSystemObject), &a, .main, block)
        }
    }
}
