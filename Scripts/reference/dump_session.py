"""A whole offline session of haru_mlx at a fixed seed, for the Swift session comparison.

    .venv-ref/bin/python -I Scripts/reference/dump_session.py --model DIR --user user.wav --voice voice.wav \
        --language en --dtype float32 --seed 0 --out .local-runs/parity/session-float32.json
"""
import argparse, json
from importlib import resources
from pathlib import Path
import mlx.core as mx
import numpy as np
from haru_mlx.audio import load_audio, prepare_reference
from haru_mlx.models import load_model
from haru_mlx.prompt import card_templates, render_card
from haru_mlx.session import DuplexConfig, DuplexSession

p = argparse.ArgumentParser()
for name in ("model", "user", "voice", "out"): p.add_argument(f"--{name}", required=True)
p.add_argument("--language", default="en"); p.add_argument("--dtype", default="float32"); p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
model = load_model(a.model, dtype=getattr(mx, a.dtype))
cards = json.loads((resources.files("haru_mlx") / "assets" / "cards" / f"{a.language}.json").read_text(encoding="utf-8"))
card = render_card(sorted(cards.items())[0][1], card_templates()["templates"][a.language])
fs = model.config.frame_samples
user = load_audio(a.user); count = max(1, -(-len(user) // fs)); user = np.pad(user, (0, count * fs - len(user)))
session = DuplexSession(model, card, prepare_reference(load_audio(a.voice)), DuplexConfig(seed=a.seed, max_frames=count + 1))
frames = [session.start()] + [session.step(user[i * fs:(i + 1) * fs]) for i in range(count - 1)]
Path(a.out).write_text(json.dumps({"prompt_length": session.prompt_length, "tokens": [f.token for f in frames],
    "text": "".join(f.text for f in frames), "audio_rms": [float(np.sqrt(np.mean(f.audio ** 2))) for f in frames]}))
print(session.prompt_length, count, "".join(f.text for f in frames).strip())
