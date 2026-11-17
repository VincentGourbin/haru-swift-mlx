// CodecStream.swift - frame-by-frame encode and decode of one conversation
// Copyright 2026 Vincent Gourbin
//
// Port of `CodecStream` in haru_mlx/models/vibevoice_codec.py (higurashi-ai/Haru @ 7fe6b76,
// Apache-2.0). `feed` takes microphone audio of any length and runs the encoder front on every whole
// front hop at once; `pop` finishes the oldest complete frame. Lives on the MLX thread.

import Foundation
import MLX

public final class CodecStream {
    let codec: VibeVoiceCodec
    var encoderState: [MLXArray] = []
    var decoderState: [MLXArray] = []
    private var pending = MLXArray.zeros([0], dtype: .float32)  // audio short of one front hop
    private var frontOut: [MLXArray] = []
    private var frontSteps = 0

    public init(_ codec: VibeVoiceCodec) {
        self.codec = codec
        reset()
    }

    public func reset() {
        (encoderState, decoderState) = codec.initState()
        pending = MLXArray.zeros([0], dtype: .float32)
        frontOut = []
        frontSteps = 0
    }

    private var stepsPerFrame: Int { codec.frameHop / codec.frontHop }

    /// Runs the encoder front over every whole front hop of new audio `[n]`.
    public func feed(_ audio: MLXArray) {
        let split = codec.frontLayers
        let all = concatenated([pending, audio.asType(.float32).reshaped([-1])])
        var hops = all.dim(0) / codec.frontHop
        var start = 0
        // Chunks of a frame, half, quarter, … keep the number of shapes small.
        var size = stepsPerFrame
        while hops > 0 {
            while size > hops { size /= 2 }
            let end = start + size * codec.frontHop
            let (out, next) = codec.runFront(all[start ..< end], Array(encoderState[..<split]))
            encoderState.replaceSubrange(..<split, with: next)
            frontOut.append(out)
            frontSteps += out.dim(0)
            start = end
            hops -= size
        }
        pending = all[start...]
    }

    /// Whole frames fed and not yet popped.
    public var framesReady: Int { frontSteps / stepsPerFrame }

    /// Finishes encoding the oldest complete frame -> latent mean `[1, 1, D]`.
    public func pop() -> MLXArray {
        let steps = stepsPerFrame, split = codec.frontLayers
        precondition(frontSteps >= steps, "no complete frame has been fed")
        let x = frontOut.count > 1 ? concatenated(frontOut, axis: 0) : frontOut[0]
        let rest = x[steps...]
        frontOut = rest.dim(0) > 0 ? [rest] : []
        frontSteps -= steps
        let (mean, next) = codec.runBack(x[..<steps], Array(encoderState[split...]))
        encoderState.replaceSubrange(split..., with: next)
        return expandedDimensions(mean, axis: 0)
    }

    /// `[1, n * 3200]` new audio -> latent means `[1, n, D]`.
    public func encode(_ audio: MLXArray) -> MLXArray {
        feed(audio)
        return concatenated((0..<framesReady).map { _ in pop() }, axis: 1)
    }

    /// `[1, n, D]` new latents -> audio `[1, n * 3200]`.
    public func decode(_ latents: MLXArray) -> MLXArray {
        let (audio, next) = codec.runDecoder(latents.reshaped([-1, latents.dim(-1)]), decoderState)
        decoderState = next
        return expandedDimensions(audio, axis: 0)
    }

    /// Every array of the stream's state (to evaluate once per frame).
    public var arrays: [MLXArray] { encoderState + decoderState + [pending] + frontOut }
}
