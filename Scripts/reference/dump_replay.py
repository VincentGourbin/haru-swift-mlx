"""A whole haru_mlx session, frame by frame, for the Swift forced replay (RealSessionReplayTests):
per frame the sampled token, the head latent, the final hidden state, the user latent and the audio.

    .venv-ref/bin/python -I Scripts/reference/dump_replay.py --model DIR --user user.wav --voice voice.wav \
        --dtype float32 --seed 5 --eop-bias 1 --out .local-runs/parity/replay-float32.safetensors

Swift replays it with the same prompt and user audio, forcing Python's token and latent at every
frame, and reports the error of its own hidden state, latent and audio against frame index.
"""
import argparse, json
from importlib import resources
import mlx.core as mx
import numpy as np
from haru_mlx.audio import load_audio, prepare_reference
from haru_mlx.models import load_model
from haru_mlx.prompt import card_templates, render_card
from haru_mlx.session import DuplexConfig, DuplexSession

p = argparse.ArgumentParser()
for name in ("model", "user", "voice", "out"): p.add_argument(f"--{name}", required=True)
p.add_argument("--dtype", default="float32"); p.add_argument("--seed", type=int, default=5)
p.add_argument("--eop-bias", type=float, default=0.0)
a = p.parse_args()
model = load_model(a.model, dtype=getattr(mx, a.dtype))
cards = json.loads((resources.files("haru_mlx") / "assets" / "cards" / "en.json").read_text(encoding="utf-8"))
card = render_card(sorted(cards.items())[0][1], card_templates()["templates"]["en"])
fs = model.config.frame_samples
user = load_audio(a.user); count = max(1, -(-len(user) // fs)); user = np.pad(user, (0, count * fs - len(user)))
reference = prepare_reference(load_audio(a.voice))
session = DuplexSession(model, card, reference, DuplexConfig(seed=a.seed, max_frames=count + 1, eop_bias=a.eop_bias))

hidden, user_latents, latents = [], [], []
emit = session._emit
def capture_emit(h, started):
    hidden.append(h.astype(mx.float32)[0])
    frame = emit(h, started)
    latents.append(session._latent.astype(mx.float32).reshape(-1))  # what is fed back next frame
    return frame
session._emit = capture_emit
sample = model.codec.sample
def capture_sample(x, key):
    z = sample(x, key); user_latents.append(z.astype(mx.float32).reshape(-1)); return z
frames = [session.start()]
model.codec.sample = capture_sample  # after the prompt: only the user latents of the steps
frames += [session.step(user[i * fs:(i + 1) * fs]) for i in range(count - 1)]
mx.save_safetensors(a.out, {
    "in.user": mx.array(user), "in.reference": mx.array(reference.astype(np.float32)),
    "in.seed": mx.array([a.seed], dtype=mx.int32), "in.eop_bias": mx.array([a.eop_bias], dtype=mx.float32),
    "tokens": mx.array([f.token for f in frames], dtype=mx.int32),
    "hidden": mx.stack(hidden), "latents": mx.stack(latents), "user_latents": mx.stack(user_latents),
    "audio": mx.array(np.stack([f.audio for f in frames]).astype(np.float32)),
})
print(a.dtype, len(frames), "frames:", "".join(f.text for f in frames).strip()[:300])
