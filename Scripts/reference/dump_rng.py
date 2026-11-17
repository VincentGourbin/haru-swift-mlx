"""MLX random draws of a session, from Python, for the Swift NoiseSource (card T-0.5).

    .venv-ref/bin/python -I Scripts/reference/dump_rng.py --out Tests/HaruTests/Fixtures/rng.json

Replays the order of draws of haru_mlx/session.py for two seeds: `_next_key` is
`key, draw = split(key)`; the head noise is `normal((1, 64)).astype(bfloat16)`; `codec.sample`
splits its key into a scale key and a noise key; `sample_token` draws with `categorical`.
Every value is written exactly (keys as uint32, floats as float32 / bfloat16 bit patterns).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np


def bits(array: mx.array) -> list[int]:
    """Bit patterns: uint32 for float32 / uint32 arrays, uint16 for bfloat16."""
    if array.dtype == mx.bfloat16:
        return np.array(array.view(mx.uint16)).reshape(-1).tolist()
    if array.dtype == mx.float32:
        return np.array(array).view(np.uint32).reshape(-1).tolist()
    return np.array(array).reshape(-1).astype(np.int64).tolist()


def replay(seed: int) -> dict:
    state = mx.random.key(seed)
    out: dict = {"seed": seed, "key": bits(state)}

    def next_key() -> mx.array:
        nonlocal state
        state, draw = mx.random.split(state)
        return draw

    draws = []
    for step in range(3):
        k = next_key()
        noise32 = mx.random.normal((1, 64), key=k)
        head = mx.random.normal((1, 64), key=next_key()).astype(mx.bfloat16)
        scale_key, noise_key = mx.random.split(next_key())
        scale = mx.random.normal((1, 1, 1), key=scale_key)
        noise = mx.random.normal((1, 3, 64), key=noise_key)
        logits = mx.array(np.linspace(-2.0, 2.0, 25, dtype=np.float32)[None] / 0.7)
        token = mx.random.categorical(logits, key=next_key())
        draws.append({
            "draw_key": bits(k), "normal_f32": bits(noise32), "head_bf16": bits(head),
            "codec_scale": bits(scale), "codec_noise": bits(noise), "categorical": bits(token),
        })
    out["draws"] = draws
    out["final_key"] = bits(state)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = {"mlx": mx.__version__, "device": str(mx.default_device()), "seeds": [replay(0), replay(1234)]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result) + "\n")
    print(f"mlx {mx.__version__} on {mx.default_device()} -> {args.out}")


if __name__ == "__main__":
    main()
