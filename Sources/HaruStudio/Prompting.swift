// Prompting.swift - the prompter's ears and brain in Haru Studio: Apple speech recognition and a local LLM
// Copyright 2026 Vincent Gourbin
//
// The listener transcribes the microphone as it comes, on the device (Speech framework, language fixed,
// no GPU: Haru's frames keep the GPU to themselves; a Voxtral reloaded per utterance delayed 8 frames in
// 436, docs/knowledge/investigations/souffleur-forcage-texte.md). At the end of each utterance the decider
// asks an OpenAI-compatible server on this Mac (gemma4-server, qwen38 serve…) what Haru should say, and
// the line goes to the Prompter, which makes Haru say it in its own voice.

import AVFoundation
import Foundation
import Speech

/// Live transcription of the user, cut utterance by utterance.
final class LiveListener: @unchecked Sendable {
    private let recognizer: SFSpeechRecognizer
    private let format = AVAudioFormat(standardFormatWithSampleRate: 48_000, channels: 1)!
    private let queue = DispatchQueue(label: "haru.studio.listener")
    private let lock = NSLock()
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?
    private var text = ""
    let onDevice: Bool

    init?(locale: Locale) {
        guard let recognizer = SFSpeechRecognizer(locale: locale), recognizer.isAvailable else { return nil }
        self.recognizer = recognizer
        onDevice = recognizer.supportsOnDeviceRecognition
    }

    static func authorize() async -> Bool {
        await withCheckedContinuation { c in
            SFSpeechRecognizer.requestAuthorization { c.resume(returning: $0 == .authorized) }
        }
    }

    func start() { queue.async { self.open() } }

    func stop() {
        queue.async {
            self.request?.endAudio()
            self.task?.cancel()
            self.request = nil
            self.task = nil
        }
    }

    /// Microphone samples at 48 kHz (audio thread): copied, then appended on the listener's queue.
    func append(_ samples: UnsafeBufferPointer<Float>) {
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)) else { return }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        buffer.floatChannelData![0].update(from: samples.baseAddress!, count: samples.count)
        queue.async { self.request?.append(buffer) }
    }

    /// The words heard since the last cut; the recognition starts over for the next utterance.
    func cut() -> String {
        let heard = lock.withLock { () -> String in let t = text; text = ""; return t }
        queue.async {
            self.request?.endAudio()
            self.open()
        }
        return heard.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func open() {
        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        request.requiresOnDeviceRecognition = onDevice
        request.addsPunctuation = true
        self.request = request
        task = recognizer.recognitionTask(with: request) { [weak self, weak request] result, _ in
            guard let self, let result, request === self.request else { return }  // a cut request's late results are dropped
            self.lock.withLock { self.text = result.bestTranscription.formattedString }
        }
    }
}

/// What Haru should say next, from an OpenAI-compatible chat server on this Mac.
final class Decider: @unchecked Sendable {
    let url: URL
    private var messages: [[String: String]]

    static let systemPrompts: [String: String] = [
        "fr": "Tu es le souffleur de Haru, une étudiante de 22 ans qui parle au téléphone avec l'utilisateur. À chaque fois que l'utilisateur a fini de parler, tu écris la phrase que Haru va dire. Réponds à ce qu'il demande, naturellement, à l'oral, en français. Une ou deux phrases courtes, 20 mots au plus, sans émoji, sans liste, sans guillemets. La transcription peut contenir des erreurs : si tu ne comprends pas, demande de répéter.",
        "en": "You are Haru's prompter. Haru is a 22-year-old student on a phone call with the user. Each time the user finishes speaking, write what Haru says next: answer what was asked, naturally, as spoken English. One or two short sentences, 20 words at most, no emoji, no list, no quotes. The transcript may contain errors: if you do not understand, ask to repeat.",
    ]

    init(url: URL, language: String) {
        self.url = url
        messages = [["role": "system", "content": Self.systemPrompts[language] ?? Self.systemPrompts["en"]!]]
    }

    /// The reply to `heard`, and the seconds it took.
    func reply(to heard: String) async throws -> (String, Double) {
        messages.append(["role": "user", "content": heard])
        var request = URLRequest(url: url, timeoutInterval: 20)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: ["messages": messages, "max_tokens": 80, "temperature": 0.7])
        let started = Date()
        let (data, _) = try await URLSession.shared.data(for: request)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let choice = (json?["choices"] as? [[String: Any]])?.first?["message"] as? [String: Any]
        guard var line = choice?["content"] as? String else { throw URLError(.cannotParseResponse) }
        line = line.trimmingCharacters(in: .whitespacesAndNewlines).trimmingCharacters(in: CharacterSet(charactersIn: "\"«» "))
        messages.append(["role": "assistant", "content": line])
        return (line, Date().timeIntervalSince(started))
    }

    /// Whether the server answers (GET /v1/models).
    static func reachable(_ url: URL) async -> Bool {
        var models = url
        while models.lastPathComponent != "v1", models.pathComponents.count > 1 { models.deleteLastPathComponent() }
        var request = URLRequest(url: models.appendingPathComponent("models"), timeoutInterval: 2)
        request.httpMethod = "GET"
        return ((try? await URLSession.shared.data(for: request))?.1 as? HTTPURLResponse)?.statusCode == 200
    }
}

/// Speech-recognition locale of a card language.
func speechLocale(_ language: String) -> Locale {
    let map = ["fr": "fr-FR", "en": "en-US", "ja": "ja-JP", "zh": "zh-CN", "ko": "ko-KR", "es": "es-ES",
               "de": "de-DE", "pt": "pt-BR", "ru": "ru-RU"]
    return Locale(identifier: map[language] ?? "en-US")
}
