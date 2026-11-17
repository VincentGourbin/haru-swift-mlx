// HaruWeights.swift - the pinned files of higurashi-ai/Haru-4B
// Copyright 2026 Vincent Gourbin
//
// Revision, sizes and SHA-256 read from the Hub on 2026-11-17 (tree API: LFS oids for the two large
// files) and from the downloaded copies (the small files). `haru download` fetches what is missing.

import Foundation

public enum HaruWeights {
    public static let repository = HaruReference.repository
    /// Commit of higurashi-ai/Haru-4B the port is checked against.
    public static let revision = "8d6e83bcd3701a76fc27a764965665da1df7da78"

    public static let config = HubFile(
        repoID: repository, revision: revision, path: "config.json", size: 2735,
        sha256: "ff29b2dab25780d33fe1ea8efc3d1364d31e7bef95b0b08514b935305ba2f427")
    public static let tokenizerConfig = HubFile(
        repoID: repository, revision: revision, path: "tokenizer_config.json", size: 496,
        sha256: "7df4f800f74978c0ede927b5938f137ec112f872059adf76ddb9918f3281668a")
    public static let tokenizer = HubFile(
        repoID: repository, revision: revision, path: "tokenizer.json", size: 19_989_325,
        sha256: "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523")
    public static let model = HubFile(
        repoID: repository, revision: revision, path: "model.safetensors", size: 10_540_113_510,
        sha256: "ad2c3534ef6bd88525e22e5da3ac4fb739368406bf3b1c852cbafbade647c0fa")

    /// Small files first, so `haru info` works as soon as possible.
    public static let all = [config, tokenizerConfig, tokenizer, model]

    /// `<models directory>/higurashi-ai/Haru-4B`.
    public static func directory(in root: URL = ModelsDirectory.resolve()) -> URL {
        root.appendingPathComponent(repository, isDirectory: true)
    }

    public struct Status: Sendable {
        public let missing: [HubFile]
        public var isComplete: Bool { missing.isEmpty }
        public var missingBytes: Int64 { missing.reduce(0) { $0 + $1.size } }
    }

    /// Files absent or of the wrong size; with `verify`, also those whose SHA-256 differs (reads 10.5 GB).
    public static func status(in root: URL = ModelsDirectory.resolve(), verify: Bool = false) throws -> Status {
        let missing = try all.filter { file in
            let url = file.localURL(in: root)
            if verify { return try !HubDownloader.isComplete(file, at: url) }
            let size = (try? FileManager.default.attributesOfItem(atPath: url.path))?[.size] as? Int64
            return size != file.size
        }
        return Status(missing: missing)
    }

    public static func download(into root: URL = ModelsDirectory.resolve(),
                                progress: @Sendable @escaping (DownloadProgress) -> Void = { _ in }) async throws {
        try await HubDownloader(root: root).download(all, progress: progress)
    }
}
