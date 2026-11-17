// WeightPlan.swift - checkpoint tensors -> model parameters, from names and shapes only
// Copyright 2026 Vincent Gourbin
//
// Port of haru_mlx/weights.py (higurashi-ai/Haru @ 7fe6b76, Apache-2.0). The checkpoint stores
// PyTorch layouts; the model stacks projections that read the same input, keeps the backbone's
// zero-centred norm gains as fp32 `1 + w`, and stores convolutions channel-last or as the matrix of
// the single matmul that evaluates them. Nothing here touches MLX: the plan is checked against the
// safetensors header before a single tensor is read.

import Foundation

public enum WeightLayout: String, Sendable {
    /// Unchanged; several sources are concatenated along axis 0.
    case copy
    /// Zero-centred norm gain -> fp32 `1 + w`.
    case gamma
    /// fp32 scalar.
    case scalar
    /// Conv1d `[out, in, k]` -> `[out, k, in]` (`loaders._convert`: transpose(0, 2, 1)).
    case channelsLast
    /// Conv1d `[out, in, k]` -> `[out, k * in]` (transpose(0, 2, 1), then reshape).
    case convMatrix
    /// ConvTranspose1d `[in, out, k]` -> `[out * k, in]` (transpose(1, 2, 0), then reshape).
    case convTransposeMatrix
}

public struct WeightConversion: Sendable, Equatable {
    public let sources: [String]
    public let layout: WeightLayout
    public let shape: [Int]
}

public enum WeightPlan {
    /// Fused backbone projections: parameter -> checkpoint projections, in row order.
    static let fused: [(target: String, parts: [String])] = [
        ("linear_attn.in_proj", ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"]),
        ("self_attn.qkv", ["q_proj", "k_proj", "v_proj"]),
        ("mlp.gate_up", ["gate_proj", "up_proj"]),
    ]

    /// `"linear_attn.in_proj_qkv"` -> (`"linear_attn.in_proj"`, 0), …
    static let fusedTarget: [String: (String, Int)] = {
        var map: [String: (String, Int)] = [:]
        for (target, parts) in fused {
            let block = target.split(separator: ".")[0]
            for (index, part) in parts.enumerated() { map["\(block).\(part)"] = (target, index) }
        }
        return map
    }()

    nonisolated(unsafe) static let projection = try! Regex(#"(lm\.layers\.\d+)\.(\w+\.\w+)\.weight"#)
    nonisolated(unsafe) static let headAda = try! Regex(#"head\.(?:layers\.(\d+)|final)\.ada\.weight"#)
    nonisolated(unsafe) static let gammaName = try! Regex(#"lm\.(?:norm|layers\.\d+\.(?:\w+_layernorm|self_attn\.[qk]_norm))\.weight"#)
    nonisolated(unsafe) static let lmConv = try! Regex(#"lm\.layers\.\d+\.linear_attn\.conv1d\.weight"#)
    nonisolated(unsafe) static let codecConv = try! Regex(#"codec\.(?:encoder|decoder)\.(?:stem|head|down\.\d+|stages\.\d+\.\d+\.conv)\.weight"#)
    nonisolated(unsafe) static let codecConvTranspose = try! Regex(#"codec\.decoder\.up\.\d+\.weight"#)

    /// `(parameter, row block)` that checkpoint tensor `name` goes to.
    static func target(_ name: String) -> (String, Int) {
        if let m = try? projection.wholeMatch(in: name),
           let layer = m.output[1].substring, let part = m.output[2].substring,
           let (target, index) = fusedTarget[String(part)] {
            return ("\(layer).\(target).weight", index)
        }
        if let m = try? headAda.wholeMatch(in: name) {  // layers in order, then the final layer
            if let layer = m.output[1].substring, let index = Int(layer) { return ("head.ada.weight", index) }
            return ("head.ada.weight", 1 << 30)
        }
        return (name, 0)
    }

    static func layout(_ name: String) -> WeightLayout {
        if name == "latent_scale" || name == "latent_bias" { return .scalar }
        if (try? gammaName.wholeMatch(in: name)) != nil { return .gamma }
        if (try? lmConv.wholeMatch(in: name)) != nil { return .channelsLast }
        if (try? codecConv.wholeMatch(in: name)) != nil { return .convMatrix }
        if (try? codecConvTranspose.wholeMatch(in: name)) != nil { return .convTransposeMatrix }
        return .copy
    }

    static func convertedShape(_ layout: WeightLayout, _ shapes: [[Int]]) throws -> [Int] {
        let first = shapes[0]
        switch layout {
        case .scalar: return []
        case .channelsLast: return [first[0], first[2], first[1]]
        case .convMatrix: return [first[0], first[2] * first[1]]
        case .convTransposeMatrix: return [first[1] * first[2], first[0]]
        case .copy, .gamma:
            if shapes.count == 1 { return first }
            guard shapes.allSatisfy({ Array($0.dropFirst()) == Array(first.dropFirst()) }) else {
                throw HaruError.weightMismatch("cannot stack tensors of shapes \(shapes)")
            }
            return [shapes.reduce(0) { $0 + $1[0] }] + first.dropFirst()
        }
    }

    /// `[parameter: conversion]` for a checkpoint's `[tensor name: shape]`; every checkpoint tensor
    /// is a source of exactly one parameter.
    public static func conversionPlan(shapes: [String: [Int]]) throws -> [String: WeightConversion] {
        var blocks: [String: [Int: String]] = [:]
        for name in shapes.keys {
            let (target, index) = target(name)
            blocks[target, default: [:]][index] = name
        }
        var plan: [String: WeightConversion] = [:]
        for (target, parts) in blocks {
            let sources = parts.keys.sorted().map { parts[$0]! }
            let layout = layout(sources[0])
            plan[target] = WeightConversion(
                sources: sources, layout: layout,
                shape: try convertedShape(layout, sources.map { shapes[$0]! })
            )
        }
        return plan
    }

    /// Throws unless `plan` produces exactly the parameters in `expected`, listing the first 20 problems.
    public static func check(_ plan: [String: WeightConversion], expected: [String: [Int]]) throws {
        var problems = Set(expected.keys).subtracting(plan.keys).sorted().map { "missing from the checkpoint: \($0)" }
        problems += Set(plan.keys).subtracting(expected.keys).sorted().map {
            "not a model parameter: \($0) (from \(plan[$0]!.sources.joined(separator: ", ")))"
        }
        problems += Set(plan.keys).intersection(expected.keys).sorted().compactMap { name in
            plan[name]!.shape == expected[name]! ? nil
                : "\(name): checkpoint gives \(plan[name]!.shape), the model needs \(expected[name]!)"
        }
        guard problems.isEmpty else {
            let shown = problems.prefix(20).joined(separator: "\n  ")
            let more = problems.count > 20 ? "\n  ... and \(problems.count - 20) more" : ""
            throw HaruError.weightMismatch("checkpoint does not match the model:\n  \(shown)\(more)")
        }
    }
}
