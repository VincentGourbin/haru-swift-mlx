// ContentView.swift - Haru Studio: card, voice, settings on the left; the live conversation on the right
// Copyright 2026 Vincent Gourbin

import HaruAudio
import HaruCore
import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @Bindable var model: StudioModel
    @State private var pickingVoice = false
    @State private var saving = false

    var body: some View {
        HSplitView {
            Form {
                Section("Character") {
                    Picker("Language", selection: $model.language) {
                        ForEach(model.library?.languages ?? [], id: \.code) { Text($0.label).tag($0.code) }
                    }
                    TextField("Name", text: field(\.name))
                    TextField("Your name", text: field(\.otherName))
                    Picker("Gender", selection: field(\.gender)) {
                        Text("—").tag(""); Text("Female").tag("female"); Text("Male").tag("male")
                    }
                    Picker("Backchannels", selection: field(\.backchannel)) {
                        Text("—").tag(""); Text("Sparse").tag("low"); Text("Moderate").tag("mid"); Text("Frequent").tag("high")
                    }
                    editor("Persona", \.persona)
                    editor("Relationship", \.relationship)
                    editor("Situation", \.situation)
                }
                Section("Voice") {
                    HStack {
                        Text(model.voiceURL?.lastPathComponent ?? "No voice").lineLimit(1).truncationMode(.middle)
                        Spacer()
                        Button("Choose…") { pickingVoice = true }
                        Button("Synthetic") { Task { await model.makeSyntheticVoice() } }
                            .help("A voice made on this Mac by macOS speech synthesis, in the card's language (free to use). Haru speaks a language better with a voice of that language.")
                    }
                    if model.voiceSeconds > 0 {
                        Text(String(format: "%.0f s used (silence trimmed, last 3 min)", model.voiceSeconds))
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    Text("Only use voices you have the right to use.").font(.caption).foregroundStyle(.secondary)
                }
                Section("Settings") {
                    Picker("Microphone", selection: $model.inputName) {
                        ForEach(model.devices.filter { $0.inputs > 0 }, id: \.name) { Text($0.name).tag($0.name) }
                    }
                    Picker("Output", selection: $model.outputName) {
                        ForEach(model.devices.filter { $0.outputs > 0 }, id: \.name) { Text($0.name).tag($0.name) }
                    }
                    HStack {
                        Text("Guidance")
                        Slider(value: $model.guidance, in: 1 ... 2, step: 0.1)
                        Text(String(format: "%.1f", model.guidance)).monospacedDigit()
                    }
                    HStack {
                        Text("Eagerness").help("Pushes Haru to start speaking sooner (0 = as trained).")
                        Slider(value: $model.eagerness, in: 0 ... 4, step: 0.5)
                        Text(String(format: "%.1f", model.eagerness)).monospacedDigit()
                    }
                    HStack {
                        Text("Temperature").help("Lower: steadier words; higher: more varied.")
                        Slider(value: $model.temperature, in: 0.3 ... 1.0, step: 0.05)
                        Text(String(format: "%.2f", model.temperature)).monospacedDigit()
                    }
                    HStack {
                        Text("Margin")
                        Slider(value: $model.marginMs, in: 5 ... 80, step: 5)
                        Text("\(Int(model.marginMs)) ms").monospacedDigit()
                    }
                    Toggle("Echo cancellation (speakers, no headphones)", isOn: $model.echoCancellation)
                    Toggle("Haru speaks first", isOn: $model.speakFirst)
                    Toggle("Keep what Haru heard (troubleshooting, local file)", isOn: $model.keepHeardAudio)
                    Toggle("Prompter: a local LLM chooses what Haru says", isOn: $model.prompterOn)
                        .help("Apple speech recognition on this Mac hears you; an OpenAI-compatible server on this Mac writes Haru's line; Haru says it in its own voice and says nothing else.")
                    if model.prompterOn {
                        TextField("LLM server", text: $model.prompterURL).font(.caption.monospaced())
                    }
                    Picker("Precision", selection: $model.precision) {
                        Text("BF16 (as published)").tag(HaruPrecision.bf16)
                        Text("8-bit (smaller, faster)").tag(HaruPrecision.int8)
                    }
                    .onChange(of: model.precision) { Task { await model.load() } }
                    .disabled(model.phase == .live)
                }
            }
            .formStyle(.grouped)
            .frame(minWidth: 360, idealWidth: 400)
            .disabled(model.phase == .live)

            ConversationPane(model: model, saving: $saving)
                .frame(minWidth: 460)
        }
        .fileImporter(isPresented: $pickingVoice, allowedContentTypes: [.audio]) { result in
            if case .success(let url) = result { model.voiceURL = url }
        }
        .fileExporter(isPresented: $saving, document: WAVPlaceholder(), contentType: .wav,
                      defaultFilename: "haru-conversation.wav") { result in
            if case .success(let url) = result { try? model.save(to: url) }
        }
    }

    private func field(_ key: WritableKeyPath<CharacterCard, String?>) -> Binding<String> {
        Binding(get: { model.card[keyPath: key] ?? "" }, set: { model.card[keyPath: key] = $0.isEmpty ? nil : $0 })
    }

    private func editor(_ title: String, _ key: WritableKeyPath<CharacterCard, String?>) -> some View {
        VStack(alignment: .leading) {
            Text(title)
            TextEditor(text: field(key)).frame(minHeight: 60).font(.body)
        }
    }
}

struct ConversationPane: View {
    @Bindable var model: StudioModel
    @Binding var saving: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                status
                Spacer()
                if model.phase == .live {
                    Button("End conversation") { model.stop() }.keyboardShortcut(.cancelAction)
                } else {
                    Button("Start conversation") { Task { await model.start() } }
                        .keyboardShortcut(.defaultAction)
                        .disabled(model.engine == nil || [.preparing].contains(model.phase) || isLoading)
                }
            }
            LevelRow(label: "You", level: model.inputLevel, tint: .blue)
            LevelRow(label: "Haru", level: model.outputLevel, tint: .orange)
            HStack {
                if let latency = model.latencyMs { Text(String(format: "Audio latency %.0f ms", latency)) }
                if model.stepMs > 0 { Text(String(format: "· frame %.0f ms / 133 ms", model.stepMs)) }
            }
            .font(.caption).foregroundStyle(.secondary).monospacedDigit()
            Lanes(user: model.userLevels, haru: model.modelLevels)
                .frame(height: 64)
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 6) {
                        if model.timeline.isEmpty {
                            Text("Who speaks when, and Haru's words, appear here.").foregroundStyle(.secondary)
                        }
                        ForEach(model.timeline) { TimelineRow(entry: $0) }
                        Color.clear.frame(height: 1).id("end")
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                .onChange(of: model.timeline) { proxy.scrollTo("end", anchor: .bottom) }
            }
            if !model.whispers.isEmpty {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Prompter").font(.caption.bold())
                    ForEach(Array(model.whispers.suffix(4).enumerated()), id: \.offset) { Text($0.element).font(.caption) }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
            }
            if !model.report.isEmpty {
                Text(model.report).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
            }
            HStack {
                Text("Wired headphones avoid echo; Bluetooth adds 100 ms or more.").font(.caption).foregroundStyle(.secondary)
                Spacer()
                Button("Save conversation…") { saving = true }.disabled(!model.canSave)
            }
        }
        .padding()
    }

    private var isLoading: Bool { if case .loading = model.phase { true } else { false } }

    @ViewBuilder private var status: some View {
        switch model.phase {
        case .loading(let message): HStack { ProgressView().controlSize(.small); Text(message) }
        case .idle: Text("Idle")
        case .preparing: HStack { ProgressView().controlSize(.small); Text("Reading the card and the voice…") }
        case .ready: Label("Ready", systemImage: "checkmark.circle.fill").foregroundStyle(.green)
        case .live: Label("Listening and speaking", systemImage: "waveform").foregroundStyle(.orange)
        case .failed(let message): Label(message, systemImage: "exclamationmark.triangle.fill").foregroundStyle(.red)
        }
    }
}

/// One stretch of speech: time, speaker, Haru's words and its reply delay.
struct TimelineRow: View {
    let entry: TimelineEntry
    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text(String(format: "%5.1f s", entry.segment.start)).monospacedDigit().foregroundStyle(.secondary).frame(width: 56, alignment: .trailing)
            if entry.speaker == .user {
                Label(String(format: "You speak (%.1f s)", entry.segment.duration), systemImage: "person.fill")
                    .foregroundStyle(.blue)
            } else {
                VStack(alignment: .leading, spacing: 2) {
                    HStack {
                        Label("Haru", systemImage: "waveform").foregroundStyle(.orange)
                        if let d = entry.delay {
                            Text(String(format: "replies %+.1f s after you stopped", d)).font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    Text(entry.text.isEmpty ? "(sound without words)" : entry.text)
                        .font(.title3).textSelection(.enabled)
                        .foregroundStyle(entry.text.isEmpty ? .secondary : .primary)
                }
            }
        }
    }
}

/// Both sides over the last 15 seconds, one bar per frame, on the same clock (newest at the right).
struct Lanes: View {
    let user: [Float]
    let haru: [Float]
    private let frames = 112  // 15 s

    var body: some View {
        Canvas { context, size in
            let end = max(user.count, haru.count)
            let start = max(0, end - frames)
            let w = size.width / CGFloat(frames)
            for (row, (levels, color)) in [(user, Color.blue), (haru, Color.orange)].enumerated() {
                let mid = size.height * (row == 0 ? 0.25 : 0.75)
                let half = size.height * 0.22
                for i in start ..< end where i < levels.count {
                    let v = CGFloat(min(1, max(0, (levels[i] + 60) / 50)))  // -60 dBFS .. -10 dBFS
                    let h = max(1, v * half)
                    let x = CGFloat(i - start) * w
                    context.fill(Path(CGRect(x: x, y: mid - h, width: max(1, w - 1), height: 2 * h)),
                                 with: .color(levels[i] > TurnDetector().thresholdDB ? color : color.opacity(0.3)))
                }
            }
        }
        .background(.quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 6))
    }
}

struct LevelRow: View {
    let label: String
    let level: Float
    let tint: Color
    var body: some View {
        HStack {
            Text(label).frame(width: 44, alignment: .leading)
            GeometryReader { g in
                ZStack(alignment: .leading) {
                    Capsule().fill(.quaternary)
                    Capsule().fill(tint).frame(width: g.size.width * CGFloat(min(1, sqrt(level) * 2)))
                }
            }
            .frame(height: 10)
        }
    }
}

/// `fileExporter` needs a document; the WAV itself is written by `StudioModel.save` to the chosen URL.
struct WAVPlaceholder: FileDocument {
    static var readableContentTypes: [UTType] { [.wav] }
    init() {}
    init(configuration: ReadConfiguration) throws {}
    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper { FileWrapper(regularFileWithContents: Data()) }
}
