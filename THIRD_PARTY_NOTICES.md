# Third-party notices

This repository is MIT-licensed (see `LICENSE`). It ports, and derives code from, the following work.

## Haru (higurashi-ai)

The model architecture, the weight layout, the session logic and, from Jalon 4, the Core Audio
client (`haru_mlx/swift/Shared/Audio.swift`: device handling, playout scheduling, WSOLA catch-up)
follow https://github.com/higurashi-ai/Haru @ `7fe6b76`.

- Copyright (c) higurashi-ai
- License: Apache License 2.0 (weights `higurashi-ai/Haru-4B` too)

Files derived from Haru carry a header naming the upstream file. The Apache-2.0 text and the NOTICE
obligations apply to those parts.

## Through Haru

- VibeVoice acoustic tokenizer architecture and weights: Copyright (c) 2025 Microsoft, MIT.
- Gated DeltaNet / Qwen3.5 model code derived from mlx-lm: Copyright (c) Apple Inc., MIT.
