// HaruCLI.swift - `haru`: command-line entry point of the Haru port
// Copyright 2026 Vincent Gourbin

import ArgumentParser
import Foundation
import HaruCore

@main
struct HaruCLI: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "haru",
        abstract: "Haru-4B full-duplex spoken dialogue on Apple Silicon (MLX).",
        subcommands: [Info.self]
    )
}

/// `haru info`: the local checkpoint's configuration and what it weighs, by part.
struct Info: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Check the local checkpoint (config.json, safetensors header) and print its inventory."
    )

    @Option(help: "Checkpoint directory (default: $HARU_MODELS_DIR/higurashi-ai/Haru-4B).")
    var model: String?

    @Flag(help: "Also check the weight conversion plan against the model's parameters.")
    var plan = false

    func run() throws {
        let directory = model.map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath, isDirectory: true) }
            ?? ModelsDirectory.resolve().appendingPathComponent(HaruReference.repository, isDirectory: true)
        let config = try HaruConfiguration.load(contentsOf: directory.appendingPathComponent("config.json"))
        let lm = config.lm
        let linear = lm.layerTypes.filter { $0 == .linearAttention }.count
        print("checkpoint   \(directory.lastPathComponent)")
        print("frame        \(config.frameSamples) samples at \(config.sampleRate) Hz = \(String(format: "%.1f", 1000 / config.frameRate)) ms")
        print("backbone     \(lm.numHiddenLayers) layers (\(linear) Gated DeltaNet, \(lm.numHiddenLayers - linear) attention), hidden \(lm.hiddenSize), vocab \(lm.vocabSize)")
        print("head         \(config.head.numLayers) layers, ffn \(config.head.ffnSize), \(config.head.numSteps) step, guidance \(config.head.guidanceScale)")

        let header = try SafetensorsHeader(contentsOf: directory.appendingPathComponent("model.safetensors"))
        let inventory = WeightInventory(header: header)
        print("")
        print("part                       tensors   parameters        bytes")
        for line in inventory.lines {
            print(line.part.rawValue.padding(toLength: 24, withPad: " ", startingAt: 0)
                + String(format: "%10d %12.1f M %10.2f GB", line.tensors, Double(line.parameters) / 1e6, Double(line.bytes) / 1e9))
        }
        print(String(format: "total                   %10d %12.1f M %10.2f GB", header.entries.count,
                     Double(inventory.totalParameters) / 1e6, Double(inventory.totalBytes) / 1e9))

        if plan {
            let conversions = try WeightPlan.conversionPlan(shapes: header.entries.mapValues(\.shape))
            try WeightPlan.check(conversions, expected: WeightPlan.parameterShapes(config: config))
            print("")
            print("PLAN OK (\(conversions.count) parameters from \(header.entries.count) tensors)")
        }
    }
}
