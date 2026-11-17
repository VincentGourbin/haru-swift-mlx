"""Expected card texts and system-prompt token ids, from haru_mlx itself (card T-0.4).

    .venv-ref/bin/python -I Scripts/reference/dump_cards.py --model DIR --out Tests/HaruTests/Fixtures/cards.json

For every language: the example card rendered with that language's template, and its token ids
(`system_prompt_ids`); plus a few hand-written strings that exercise the tokenizer (special tokens,
accents, CJK, emoji, newlines, an empty prompt). Only Apache-2.0 upstream text and token ids are written.
"""

from __future__ import annotations

import argparse
import json
from importlib import resources
from pathlib import Path

from tokenizers import Tokenizer

from haru_mlx.prompt import card_templates, render_card, system_prompt_ids

EXTRA = [
    "",
    "You are Haru.",
    "Tu t'appelles Haru, tu parles français : « déjà vu », naïve, cœur.",
    "あなたは春です。東京で働いています。",
    "你好，我叫小春。",
    "안녕하세요, 저는 하루예요.",
    "Emoji 🙂👍 and  double  spaces\n\nand a blank line.",
    "Ends with a newline\n",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(str(Path(args.model) / "tokenizer.json"))
    templates = card_templates()
    root = resources.files("haru_mlx") / "assets" / "cards"
    cards = []
    for entry in templates["languages"]:
        code = entry["code"]
        presets = json.loads((root / f"{code}.json").read_text(encoding="utf-8"))
        for preset, card in presets.items():
            text = render_card(card, templates["templates"][code])
            cards.append({"language": code, "preset": preset, "text": text, "ids": system_prompt_ids(tokenizer, text)})
    extra = [{"text": text, "ids": system_prompt_ids(tokenizer, text)} for text in EXTRA]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"cards": cards, "extra": extra}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{len(cards)} cards, {len(extra)} extra strings -> {args.out}")


if __name__ == "__main__":
    main()
