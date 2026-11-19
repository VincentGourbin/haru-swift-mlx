"""Offline prompter loop: a recording of the user -> what a decision model would whisper to Haru, and when.

    plan   USER.wav PLAN.jsonl [LANG] [ORACLE.txt]   find the user's utterances, transcribe each one as soon as it ends
                                 (Voxtral Realtime), ask a local LLM what Haru should say, and write
                                 {"at", "text"} lines for `haru run --steer-plan`. Causal: a line is due at
                                 the end of the utterance plus the measured ASR and LLM times, and the LLM
                                 only sees what was said up to that utterance.
    judge  STEREO.wav PLAN.jsonl Voxtral on Haru's channel after each line: word recall of the line, and
                                 the delay from the end of the user's utterance to Haru's voice.

Environment: VOXTRAL_CLI (VoxtralCLI binary), VOXTRAL_MODEL (default realtime-4b-6bit), PROMPTER_URL (an
OpenAI-compatible server, default http://127.0.0.1:8089/v1/chat/completions). Standard library only.
Run with `python3 -I`. Recordings and plans stay under .local-runs/ (never committed).
"""
import array, json, math, os, re, subprocess, sys, tempfile, time, urllib.request, wave

SR, FS = 24000, 3200
FRAME = FS / SR
CLI = os.environ.get("VOXTRAL_CLI", "VoxtralCLI")
ASR_MODEL = os.environ.get("VOXTRAL_MODEL", "realtime-4b-6bit")
URL = os.environ.get("PROMPTER_URL", "http://127.0.0.1:8089/v1/chat/completions")

SYSTEM = {
    "fr": ("Tu es Haru, une étudiante de 22 ans, chaleureuse et curieuse, au téléphone avec l'utilisateur. Tu écris exactement ce que tu dis à voix haute, à la première personne, en français parlé. Réponds à ce qu'il demande, naturellement : une ou deux phrases courtes, 20 mots au plus, sans émoji, sans liste, sans guillemets, sans didascalie. Ne dis jamais que tu es une IA, un modèle ou un souffleur. Ce que tu entends est une transcription automatique qui peut contenir des erreurs : si tu ne comprends pas, demande de répéter."),
    "en": ('You are Haru, a warm and curious 22-year-old student, on a phone call with the user. Write exactly what you say out loud, in the first person, as spoken English. Answer what was asked, naturally: one or two short sentences, 20 words at most, no emoji, no list, no quotes, no stage directions. Never say you are an AI, a model or a prompter. What you hear is an automatic transcript that may contain errors: if you do not understand, ask to repeat.'),
}


def read_wav(path, channel=None):
    w = wave.open(path)
    n, ch = w.getnframes(), w.getnchannels()
    assert w.getframerate() == SR and w.getsampwidth() == 2, "24 kHz 16-bit WAV expected"
    a = array.array("h", w.readframes(n))
    w.close()
    if ch > 1:
        a = a[(channel or 0)::ch]
    return a


def write_wav(path, samples):
    o = wave.open(path, "wb")
    o.setnchannels(1); o.setsampwidth(2); o.setframerate(SR)
    o.writeframes(samples.tobytes()); o.close()


def levels(a):
    out = []
    for k in range(0, len(a), FS):
        x = a[k:k + FS]
        out.append(10 * math.log10(max(sum(v * v for v in x) / max(len(x), 1) / 32768 ** 2, 1e-12)))
    return out


def segments(lv, threshold=-40, hang=5, minimum=2):
    """Same rule as HaruCore.TurnDetector: (start, end) in seconds."""
    out, start, last = [], None, 0
    for i, l in enumerate(lv):
        if l > threshold:
            start = i if start is None else start
            last = i
        elif start is not None and i - last >= hang:
            if last - start + 1 >= minimum: out.append((start * FRAME, (last + 1) * FRAME))
            start = None
    if start is not None and last - start + 1 >= minimum: out.append((start * FRAME, (last + 1) * FRAME))
    return out


def transcribe(samples, tail=1.0):
    """Voxtral Realtime lags its audio (--delay 960 ms): `tail` seconds of silence let it finish the words."""
    samples = samples + array.array("h", bytes(2 * int(tail * SR)))
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name
    write_wav(path, samples)
    out = subprocess.run([CLI, "realtime", path, "-m", ASR_MODEL, "--delay", "960"], capture_output=True, text=True).stdout
    os.unlink(path)
    lines = out.splitlines()
    text = next((lines[i + 1] for i, l in enumerate(lines) if l.startswith("-----") and i + 1 < len(lines)), "").strip()
    t = re.search(r"Time:\s*([\d.]+)s", out)
    return text, float(t.group(1)) if t else 0.0


def ask(messages):
    body = json.dumps({"messages": messages, "max_tokens": 80, "temperature": 0.7}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    started = time.time()
    with urllib.request.urlopen(req, timeout=60) as r:
        reply = json.load(r)["choices"][0]["message"]["content"]
    return reply.strip().strip('"«» '), time.time() - started


def plan(user_path, plan_path, language="fr", oracle=None):
    """`oracle`: a text file with the true words of each utterance, one per line, used instead of the ASR
    text (the ASR still runs, for its time): separates ASR errors from the rest of the loop."""
    truth = [l.strip() for l in open(oracle) if l.strip()] if oracle else None
    user = read_wav(user_path)
    messages = [{"role": "system", "content": SYSTEM[language]}]
    ask(messages + [{"role": "user", "content": "(test)"}])  # warm the server: a live session keeps it loaded
    rows = []
    for n, (start, end) in enumerate(segments(levels(user))):
        heard, asr_s = transcribe(user[int(max(0, start - 1.0) * SR):int(end * SR)])
        if truth: heard = truth[n] if n < len(truth) else heard
        if not heard:
            continue
        messages.append({"role": "user", "content": heard})
        line, llm_s = ask(messages)
        messages.append({"role": "assistant", "content": line})
        at = end + asr_s + llm_s
        rows.append({"at": round(at, 2), "text": line, "user": heard, "user_end": round(end, 2),
                     "asr_s": round(asr_s, 2), "llm_s": round(llm_s, 2)})
        print(f"{start:6.2f}-{end:6.2f} s  heard «{heard}»  -> «{line}»  (ASR {asr_s:.2f} s, LLM {llm_s:.2f} s, due {at:.2f} s)")
    with open(plan_path, "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")


def norm(t):
    return re.sub(r"[^\w' ]", " ", t.lower().replace("’", "'")).split()


def lcs(a, b):
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a)):
        for j in range(len(b)):
            d[i + 1][j + 1] = d[i][j] + 1 if a[i] == b[j] else max(d[i][j + 1], d[i + 1][j])
    return d[-1][-1]


def judge(stereo_path, plan_path):
    haru = read_wav(stereo_path, channel=1)
    lv = levels(haru)
    rows = [json.loads(l) for l in open(plan_path) if l.strip()]
    recalls, delays = [], []
    for k, r in enumerate(rows):
        start = r["at"] - 0.3
        stop = min(rows[k + 1]["user_end"] if k + 1 < len(rows) else len(haru) / SR, start + 2.5 + 0.45 * len(norm(r["text"])))
        heard, _ = transcribe(haru[int(start * SR):int(stop * SR)])
        rec = lcs(norm(r["text"]), norm(heard)) / max(1, len(norm(r["text"])))
        onset = next((i * FRAME for i in range(int(r["at"] / FRAME), int(stop / FRAME)) if i < len(lv) and lv[i] > -40), None)
        delay = None if onset is None else onset - r["user_end"]
        recalls.append(rec)
        if delay is not None and rec >= 0.5: delays.append(delay)
        print(f"«{r['user']}» -> «{r['text']}»\n    Haru said «{heard}»  recall {rec:.0%}  voice {'-' if delay is None else f'{delay:+.2f} s'} after the user")
    d = sorted(delays)
    print(f"mean recall {sum(recalls) / max(1, len(recalls)):.0%}, lines said (recall >= 50 %) {len(delays)}/{len(rows)}, "
          f"median delay {d[len(d) // 2]:+.2f} s" if d else f"mean recall {sum(recalls) / max(1, len(recalls)):.0%}, no line said")


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:]
    {"plan": plan, "judge": judge}[cmd](*rest)
