"""An offline haru_mlx session written as the Swift `haru run --stereo --transcript` writes it
(stereo WAV, user left and model right; one JSON line per frame), to compare turn timing.

    .venv-ref/bin/python -I Scripts/reference/dump_turns.py --model DIR --user user.wav --voice voice.wav \
        --dtype bfloat16 --seed 0 --wav out.wav --transcript out.jsonl
"""
import argparse, json, wave
from importlib import resources
import mlx.core as mx
import numpy as np
from haru_mlx.audio import load_audio, prepare_reference
from haru_mlx.models import load_model
from haru_mlx.prompt import card_templates, render_card
from haru_mlx.session import DuplexConfig, DuplexSession

p = argparse.ArgumentParser()
for name in ("model", "user", "wav", "transcript"): p.add_argument(f"--{name}", required=True)
p.add_argument("--voice", default=None)
p.add_argument("--language", default="en"); p.add_argument("--dtype", default="bfloat16"); p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
model = load_model(a.model, dtype=getattr(mx, a.dtype))
cards = json.loads((resources.files("haru_mlx") / "assets" / "cards" / f"{a.language}.json").read_text(encoding="utf-8"))
card = render_card(sorted(cards.items())[0][1], card_templates()["templates"][a.language])
fs = model.config.frame_samples
user = load_audio(a.user); count = max(1, -(-len(user) // fs)); user = np.pad(user, (0, count * fs - len(user)))
session = DuplexSession(model, card, prepare_reference(load_audio(a.voice)) if a.voice else None, DuplexConfig(seed=a.seed, max_frames=count + 1))
frames = [session.start()] + [session.step(user[i * fs:(i + 1) * fs]) for i in range(count - 1)]
reply = np.concatenate([f.audio for f in frames])
pcm = (np.clip(np.stack([user, reply], 1), -1, 1) * 32767).astype("<i2")
with wave.open(a.wav, "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(24000); w.writeframes(pcm.tobytes())
with open(a.transcript, "w") as f:
    for i, fr in enumerate(frames):
        f.write(json.dumps({"frame": i, "token": int(fr.token), "text": fr.text, "time": i * fs / 24000}) + "\n")
print("".join(fr.text for fr in frames).strip())
