# Haru MLX Swift

Real-time, full-duplex spoken conversation with [Haru-4B](https://huggingface.co/higurashi-ai/Haru-4B)
(higurashi-ai, Apache-2.0) on Apple Silicon, **100 % Swift and in-process** with
[MLX Swift](https://github.com/ml-explore/mlx-swift).

Haru listens and speaks at the same time. Every 133 ms it takes one frame of your audio and
produces one frame of its own speech and text: your speech latent, its previous speech latent and
its previous text token are summed into one position of a Qwen3.5-4B backbone (24 Gated DeltaNet +
8 attention layers); a one-step MeanFlow head produces the next 64-dim audio latent, and a causal
VibeVoice codec turns it into 24 kHz audio.

Upstream ships a Python MLX package and a macOS app that runs it as a server. This port runs the
whole model inside the app, so it can be embedded in other apps and, later, studied on iPhone.

## Status

| Part | Status |
|---|---|
| Configuration, checkpoint inventory (`haru info`) | ✅ |
| Weight conversion plan | 🔜 Jalon 0 |
| VibeVoice codec (streaming) | 🔜 Jalon 1 |
| Qwen3.5 backbone, Haru model | 🔜 Jalon 2 |
| MeanFlow head, prompt, session, offline wav → wav | 🔜 Jalon 3 |
| Live conversation (Core Audio, in-process) | 🔜 Jalon 4 |
| 8-bit, reference profiles, endurance | 🔜 Jalon 5 |
| App, API | 🔜 Jalon 6 |

The porting plan is in [`PLAN.md`](PLAN.md) (French).

## Build

```bash
Scripts/check-build.sh                      # xcodebuild Release (not `swift build`: MLX's metallib)
.xcodebuild/Build/Products/Release/haru info
```

Weights live in `$HARU_MODELS_DIR/higurashi-ai/Haru-4B` (default `~/Library/Caches/models`).

## Notes

Haru-4B is a pretrained base model released for research: it holds a natural conversation but is
not a reliable source of facts. It can imitate the voice of a reference clip; only use voices you
have the right to use. No voice clip ships with this repository.

## License

Code: MIT (`LICENSE`). Parts derived from higurashi-ai/Haru are Apache-2.0; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
