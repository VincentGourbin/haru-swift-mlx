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
| Configuration, weights (`haru info`, `haru download`) | ✅ pinned revision, SHA-256 |
| VibeVoice codec (streaming) | ✅ bf16 bit-exact with haru_mlx |
| Qwen3.5 backbone, MeanFlow head | ✅ fp32 parity ~1e-5, same top-1 in bf16 |
| Session, offline wav → wav (`haru run`) | ✅ |
| Live conversation (`haru talk`, Core Audio in-process) | ✅ step 44.3 ms (bf16, p99 44.9 over 5 min); reply latency 65 ms median on a virtual device (real mic/headphones add their own latency) |
| Test app (`Haru Studio`) | 🧪 first conversations held; timeline of turns, eagerness and temperature controls |
| 8-bit (`--precision int8`) | ✅ step 32.9 ms; twice bf16's distance to fp32 over a 410-frame forced replay — an option, not the default |
| Turn timeline (`haru turns`, app) | ✅ reply delay per turn; turn timing matches haru_mlx over 10 seeds |
| Compiled per-layer steps, reference profiles | 🔜 Jalon 5 |

The porting plan is in [`PLAN.md`](PLAN.md) (French).

## Build and run

```bash
Scripts/check-build.sh                      # xcodebuild Release (not `swift build`: MLX's metallib)
H=.xcodebuild/Build/Products/Release/haru
$H download                                 # 10.5 GB, resumable, SHA-256 checked
$H run user.wav reply.wav --voice voice.wav --stereo      # offline
$H talk --voice voice.wav                   # live, wired headphones recommended
Scripts/package-app.sh && open ".build/Haru Studio.app"   # the app
```

Weights live in `$HARU_MODELS_DIR/higurashi-ai/Haru-4B` (default `~/Library/Caches/models`).

## Notes

Haru-4B is a pretrained base model released for research: it holds a natural conversation but is
not a reliable source of facts. It can imitate the voice of a reference clip; only use voices you
have the right to use. No voice clip ships with this repository.

## License

Code: MIT (`LICENSE`). Parts derived from higurashi-ai/Haru are Apache-2.0; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
