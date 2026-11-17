// HaruWeightsTests.swift - the pinned Haru-4B files
// Copyright 2026 Vincent Gourbin

import Foundation
import Testing
@testable import HaruCore

@Suite struct HaruWeightsTests {
    @Test func filesArePinnedToOneRevision() {
        #expect(HaruWeights.all.count == 4)
        #expect(HaruWeights.all.allSatisfy { $0.revision == HaruWeights.revision && $0.sha256?.count == 64 })
        #expect(HaruWeights.model.url.absoluteString
            == "https://huggingface.co/higurashi-ai/Haru-4B/resolve/\(HaruWeights.revision)/model.safetensors")
    }

    @Test func emptyDirectoryMissesEverything() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("haru-empty-\(UUID().uuidString)")
        let status = try HaruWeights.status(in: root)
        #expect(status.missing.count == 4)
        #expect(status.missingBytes == 2735 + 496 + 19_989_325 + 10_540_113_510)
    }
}

/// The downloaded copies against the pinned SHA-256 (gated on HARU_MODELS_DIR; reads 10.5 GB).
@Suite(.enabled(if: ProcessInfo.processInfo.environment["HARU_MODELS_DIR"] != nil))
struct RealWeightsTests {
    @Test func localCopiesMatchThePins() throws {
        let status = try HaruWeights.status(verify: true)
        #expect(status.isComplete, "mismatching: \(status.missing.map(\.path))")
    }
}
