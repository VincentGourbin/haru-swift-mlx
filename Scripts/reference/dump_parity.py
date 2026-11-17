"""Reference tensors of haru_mlx on the real checkpoint, for the Swift parity tests.

    .venv-ref/bin/python -I Scripts/reference/dump_parity.py --model DIR --dtype float32|bfloat16 \
        --out .local-runs/parity/haru-<dtype>.safetensors

Fixed inputs as in upstream's tests/parity/dump_reference.py (card "You are Haru.", 24 reference
positions, 6 single steps, a clip that ends inside a frame), run through haru_mlx itself. Real-weight
activations stay under .local-runs (never committed).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np

from haru_mlx.models import load_model
from haru_mlx.models.vibevoice_codec import CodecStream

PROMPT = "You are Haru."
PREFILL_AUDIO = 24
STEPS = 6
CLIP_SAMPLES = 5 * 3200 + 1234


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    dtype = getattr(mx, args.dtype)
    model = load_model(args.model, dtype=dtype)
    cfg = model.config
    rng = np.random.default_rng(0)
    out: dict[str, mx.array] = {}

    # ---------------------------------------------------------------- inputs
    card = model.tokenizer.encode(f"<|im_start|>system\n{PROMPT}<|im_end|>\n", add_special_tokens=False).ids
    n_card, n = len(card), len(card) + PREFILL_AUDIO + STEPS
    text = rng.integers(1000, 5000, size=n)
    text[:n_card] = card
    text[n_card : n_card + PREFILL_AUDIO] = cfg.pad_token_id
    model_latents = rng.standard_normal((n, cfg.latent_dim)).astype(np.float32) * 2.0
    user_latents = rng.standard_normal((n, cfg.latent_dim)).astype(np.float32) * 2.0
    positions = np.arange(n)
    model_mask = positions >= n_card
    user_mask = positions >= n_card + PREFILL_AUDIO
    noise = rng.standard_normal((n, cfg.latent_dim)).astype(np.float32)
    clip = (0.1 * rng.standard_normal(CLIP_SAMPLES)).astype(np.float32)
    frames = (0.1 * rng.standard_normal(3 * cfg.frame_samples)).astype(np.float32)
    latents = rng.standard_normal((1, 3, cfg.latent_dim)).astype(np.float32) * 2.0
    for name, value in dict(text=text.astype(np.int32), model_latents=model_latents, user_latents=user_latents,
                            model_mask=model_mask.astype(np.int32), user_mask=user_mask.astype(np.int32),
                            noise=noise, clip=clip, frames=frames, latents=latents).items():
        out[f"in.{name}"] = mx.array(value)
    out["in.prefill"] = mx.array([n - STEPS], dtype=mx.int32)

    # ----------------------------------------------------------- backbone
    cache = model.new_cache()
    prefill = n - STEPS
    spans = [(0, prefill)] + [(i, i + 1) for i in range(prefill, n)]
    hidden = []
    for index, (s, e) in enumerate(spans):
        emb = model.embed(
            mx.array(text[None, s:e]), mx.array(model_latents[None, s:e]), mx.array(user_latents[None, s:e]),
            mx.array(model_mask[None, s:e]), mx.array(user_mask[None, s:e]),
        )
        if index == 0:
            out["lm.embed"] = emb[0]
            h = emb
            for i, (layer, layer_cache) in enumerate(zip(model.lm.layers, cache)):
                h = layer(h, layer_cache)
                out[f"lm.layer.{i}"] = h[0]
            h = model.lm.norm(h)
        else:
            h = model(emb, cache)
        hidden.append(h[0])
        mx.eval(hidden[-1], [a for c in cache for a in c.arrays()])
    hidden = mx.concatenate(hidden)
    out["lm.hidden"] = hidden
    stream_ids = mx.array([cfg.pad_token_id, cfg.eop_token_id])
    out["lm.logits"] = model.logits(hidden[-STEPS:])
    out["lm.logits_stream_fp32"] = model.logits_fp32(hidden[-STEPS:], stream_ids)

    # ---------------------------------------------------------------- head
    out["head.latents"] = model.head.sample(hidden, mx.array(noise).astype(model.dtype), cfg.head.guidance_scale)

    # --------------------------------------------------------------- codec
    out["codec.encode_clip"] = model.codec.encode(mx.array(clip))[0]
    out["codec.decode_latents"] = model.codec.decode(mx.array(latents))[0]
    stream = CodecStream(model.codec)
    out["codec.stream_encode"] = stream.encode(mx.array(frames)[None])[0]
    out["codec.stream_decode"] = stream.decode(mx.array(latents))[0]
    fed = CodecStream(model.codec)
    for start, end in ((0, 1000), (1000, 5300), (5300, 9600)):  # uneven blocks
        fed.feed(mx.array(frames[start:end]))
    out["codec.feed_encode"] = mx.concatenate([fed.pop() for _ in range(fed.frames_ready)], axis=1)[0]

    out = {k: (v.astype(mx.float32) if v.dtype == mx.bfloat16 else v) for k, v in out.items()}
    mx.eval(out)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(args.out, out)
    print(f"{len(out)} tensors -> {args.out} (prefill {prefill}, {STEPS} steps, dtype {args.dtype})")


if __name__ == "__main__":
    main()
