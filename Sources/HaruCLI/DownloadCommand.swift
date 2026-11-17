// DownloadCommand.swift - `haru download`: the pinned files of higurashi-ai/Haru-4B (resumable, SHA-256 checked)
// Copyright 2026 Vincent Gourbin
// Adapted from sperid-swift-mlx `idol download`.

import ArgumentParser
import Foundation
import HaruCore

struct Download: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Download the pinned Haru-4B files (10.5 GB), or check the local copies with --verify.")

    @Option(help: "Models directory (default: $HARU_MODELS_DIR, else ~/Library/Caches/models).")
    var modelsDir: String?

    @Flag(help: "Check the SHA-256 of every local file (reads 10.5 GB) instead of sizes only.")
    var verify = false

    func run() async throws {
        let root = modelsDir.map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath) } ?? ModelsDirectory.resolve()
        let status = try HaruWeights.status(in: root, verify: verify)
        let checked = verify ? "SHA-256 verified" : "sizes checked"
        guard !status.isComplete else {
            print("DOWNLOAD OK (\(HaruWeights.all.count) files present, \(checked), revision \(HaruWeights.revision.prefix(8)))")
            return
        }
        print(String(format: "downloading %d files (%.2f GB) into %@", status.missing.count,
                     Double(status.missingBytes) / 1e9, HaruWeights.directory(in: root).path))
        let reporter = ProgressReporter()
        try await HubDownloader(root: root).download(status.missing) { reporter.report($0) }
        print("DOWNLOAD OK (\(status.missing.count) files downloaded and SHA-256 verified)")
    }
}

/// Prints one line per file at every 10 % (stdout stays readable in logs).
final class ProgressReporter: @unchecked Sendable {
    private var lastDecile: [String: Int] = [:]
    private let lock = NSLock()

    func report(_ p: DownloadProgress) {
        guard p.totalBytes > 0 else { return }
        let decile = Int(10 * p.writtenBytes / p.totalBytes)
        lock.lock(); defer { lock.unlock() }
        guard decile != lastDecile[p.file] else { return }
        lastDecile[p.file] = decile
        print(String(format: "  [%d/%d] %@ %3d %%", p.fileIndex + 1, p.fileCount, p.file, decile * 10))
    }
}
