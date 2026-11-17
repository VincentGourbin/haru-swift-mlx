// swift-tools-version: 6.4
// Package.swift - Haru-4B (higurashi-ai) real-time full-duplex spoken dialogue for Apple Silicon (MLX)
// Copyright 2026 Vincent Gourbin

import PackageDescription

let package = Package(
    name: "HaruSwift",
    platforms: [.macOS(.v15), .iOS(.v27)],
    products: [
        // MARK: - Libraries
        /// Configuration, weights, models (backbone, codec, MeanFlow head) and the duplex session.
        .library(name: "HaruCore", targets: ["HaruCore"]),
        // HaruAudio (Core Audio duplex I/O, playout) lands in Jalon 4, HaruUI / haru-studio in Jalon 6.
        // MARK: - CLI Tools
        .executable(name: "haru", targets: ["HaruCLI"]),
    ],
    dependencies: [
        // Same validated range as sperid-swift-mlx: host apps (Fluxforge Studio) pin 0.31.6 exact.
        // Widen the upper bound only after the full real-weight run.
        .package(url: "https://github.com/ml-explore/mlx-swift", "0.31.6"..<"0.33.0"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.8.2"),
        .package(url: "https://github.com/VincentGourbin/swift-mlx-profiler", from: "1.5.1"),
        // Tokenizers only (tokenizer.json of the checkpoint), from Jalon 0 card T-0.4.
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.3"),
    ],
    targets: [
        // MARK: - Libraries
        .target(
            name: "HaruCore",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "MLXProfiler", package: "swift-mlx-profiler"),
            ]
        ),
        // MARK: - CLI Tools
        .executableTarget(
            name: "HaruCLI",
            dependencies: [
                "HaruCore",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
        // MARK: - Tests
        .testTarget(
            name: "HaruTests",
            dependencies: ["HaruCore"]
        ),
    ]
)
