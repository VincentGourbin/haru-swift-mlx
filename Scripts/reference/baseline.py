"""Baseline step time of the upstream haru_mlx on this Mac (card T-0.2).

    .venv-ref/bin/python -I Scripts/reference/baseline.py --model DIR [--user user.wav] [--voice voice.wav]
        [--quantize 8] [--cadence live|burst] [--warm-ms 30] [--feed] [--seconds 60]
        [--out .local-runs/bench.noindex/x.json]

`live` feeds one user frame every 133 ms on the wall clock, as a microphone would, so the GPU idles
between frames the way it does in a real conversation; `burst` steps back to back (what
`run_inference` measures). `--warm-ms` busy-polls that long before each deadline instead of
sleeping (upstream worker.py), to separate CPU wake-up from the GPU clock. `--feed` hands the user
audio over in 64-sample blocks on the wall clock and calls `step()` without audio, as the live Swift
path will (the encoder front is then spread over the frame). Without `--user`, the user side is low-level noise. Prints p50/p95/p99/max
of `frame.seconds` and the process peak memory, and writes them as JSON.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np

from haru_mlx import DuplexConfig, DuplexSession, HaruModel
from haru_mlx.audio import load_audio, prepare_reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--user")
    parser.add_argument("--voice")
    parser.add_argument("--quantize", type=int, choices=(8,), default=None)
    parser.add_argument("--cadence", choices=("live", "burst"), default="live")
    parser.add_argument("--warm-ms", type=float, default=0.0)
    parser.add_argument("--feed", action="store_true")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--warmup-frames", type=int, default=16)
    parser.add_argument("--out")
    args = parser.parse_args()

    import mlx.core as mx

    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    mx.set_cache_limit(1 << 30)  # as LocalBackend does
    started = time.perf_counter()
    model = HaruModel.from_pretrained(args.model, quantize=args.quantize)
    load_seconds = time.perf_counter() - started
    rate, frame = model.config.sample_rate, model.config.frame_samples
    frames = int(args.seconds * rate / frame)

    if args.user:
        user = load_audio(args.user)
    else:
        user = (0.003 * np.random.default_rng(0).standard_normal(frames * frame)).astype(np.float32)
    user = np.pad(user, (0, max(0, frames * frame - len(user))))[: frames * frame]
    voice = prepare_reference(load_audio(args.voice)) if args.voice else None

    session = DuplexSession(model, system_prompt="You are Haru.", reference=voice, config=DuplexConfig(seed=0))
    prompt_seconds = session.start().seconds
    period = frame / rate
    warm = args.warm_ms / 1000

    def wait_until(deadline: float) -> None:
        delay = deadline - time.perf_counter() - warm
        if delay > 0:
            time.sleep(delay)
        while time.perf_counter() < deadline:  # busy-poll the last `warm` seconds
            pass

    block = 64
    times = []
    due = time.perf_counter()
    for k in range(frames):
        audio = user[k * frame : (k + 1) * frame]
        if args.feed:
            for j in range(0, frame, block):
                if args.cadence == "live":
                    wait_until(due + (j + block) / rate)
                session.feed(audio[j : j + block])
            due += period
            times.append(session.step().seconds)
            continue
        if args.cadence == "live":
            due += period
            wait_until(due)
        times.append(session.step(audio).seconds)

    ms = 1000 * np.array(times[args.warmup_frames :])
    result = {
        "quantize": args.quantize, "cadence": args.cadence, "warm_ms": args.warm_ms, "feed": args.feed, "frames": len(ms),
        "load_s": round(load_seconds, 2), "prompt_s": round(prompt_seconds, 3),
        "p50_ms": round(float(np.percentile(ms, 50)), 2), "p95_ms": round(float(np.percentile(ms, 95)), 2),
        "p99_ms": round(float(np.percentile(ms, 99)), 2), "max_ms": round(float(ms.max()), 2),
        "late_frames": int((ms > 1000 * period).sum()),
        "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2),  # bytes on macOS
        "mlx_peak_gb": round(mx.get_peak_memory() / 1e9, 2), "mlx": mx.__version__,
    }
    print(json.dumps(result))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
