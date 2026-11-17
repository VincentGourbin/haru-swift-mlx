// HaruError.swift - errors of the Haru port
// Copyright 2026 Vincent Gourbin

import Foundation

public enum HaruError: Error, LocalizedError, Equatable {
    /// A weight file is missing from the local models directory.
    case missingFile(String)
    /// A safetensors file could not be parsed (truncated, bad header).
    case invalidSafetensors(String)
    /// A weight file does not match its contract (names, shapes, dtypes) or its SHA-256.
    case weightMismatch(String)
    /// `config.json` is not a Haru checkpoint or is inconsistent.
    case invalidConfiguration(String)
    /// A request parameter is outside the range the reference implementation accepts.
    case invalidRequest(String)

    public var errorDescription: String? {
        switch self {
        case .missingFile(let m): "missing file: \(m)"
        case .invalidSafetensors(let m): "invalid safetensors: \(m)"
        case .weightMismatch(let m): "weight mismatch: \(m)"
        case .invalidConfiguration(let m): "invalid configuration: \(m)"
        case .invalidRequest(let m): "invalid request: \(m)"
        }
    }
}
