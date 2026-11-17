// MLXWorker.swift - the single long-lived thread that runs every MLX call
// Copyright 2026 Vincent Gourbin
//
// MLX keeps its GPU streams per thread, and lazy arrays built on one thread must not be evaluated on
// another (sperid pitfall "plantage MLX entre threads"; upstream haru_mlx/worker.py). A serial
// DispatchQueue orders blocks but does not pin a thread: this is a real Thread with a job queue.

import Foundation

public final class MLXWorker: @unchecked Sendable {
    private let condition = NSCondition()
    private var jobs: [() -> Void] = []
    private var thread: Thread!

    public init(name: String = "haru-mlx") {
        thread = Thread { [unowned self] in self.loop() }
        thread.name = name
        thread.qualityOfService = .userInteractive
        thread.stackSize = 16 << 20
        thread.start()
    }

    /// True on the worker thread.
    public var isCurrent: Bool { Thread.current === thread }

    public func submit(_ job: @escaping () -> Void) {
        condition.lock()
        jobs.append(job)
        condition.signal()
        condition.unlock()
    }

    /// Runs `body` on the worker thread and returns its result.
    public func run<T>(_ body: @escaping () throws -> T) async throws -> T {
        try await withCheckedThrowingContinuation { continuation in
            submit {
                do { continuation.resume(returning: try body()) } catch { continuation.resume(throwing: error) }
            }
        }
    }

    private func loop() {
        while true {
            condition.lock()
            while jobs.isEmpty { condition.wait() }
            let job = jobs.removeFirst()
            condition.unlock()
            autoreleasepool { job() }
        }
    }
}
