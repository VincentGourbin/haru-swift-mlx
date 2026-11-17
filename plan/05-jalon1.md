# 5. Jalon 1 — Codec VibeVoice (audio ↔ latents), en flux

Pourquoi d'abord : c'est la partie la plus éloignée de ce qu'on sait déjà faire (le backbone a un
équivalent Swift dans mlx-swift-lm), et le seul composant testable à l'oreille seul.

| Fiche | Contenu | Porte |
|---|---|---|
| T-1.1 | Fixtures tiny : `Scripts/reference/make_tiny.py` — le générateur amont exige torch **et** le paquet PyTorch `haru` ; le nôtre tire des tenseurs aléatoires (numpy, graine fixe) en disposition checkpoint, noms et formes déduits de `tiny_config.json` par l'inverse du plan de T-0.3 — puis `dump_codec.py` qui fait tourner **haru_mlx** ; `check-fixtures.sh`, `check-parity.sh` | `FIXTURES OK` |
| T-1.2 | `CausalConv1d`, `CausalConvTranspose1d` (disposition matrice) + tests de forme d'état | `PARITY OK conv` ≤ 1e-6 fp32 |
| T-1.3 | Mélangeur depthwise : noyau Metal amont (`MLXFast.metalKernel`, source copiée) + repli tableaux ; `Block` | noyau = repli ≤ 1e-6 ; `PARITY OK block` |
| T-1.4 | `Encoder`, `Decoder`, `VibeVoiceCodec.encode/decode` clip entier | `PARITY OK codec` tiny ≤ 1e-5 fp32 |
| T-1.5 | `CodecStream` : `feed` par blocs quelconques (8/4/2/1 sauts), `pop`, `decode` trame par trame ; trois fonctions compilées | flux = clip entier (≤ 1e-6) pour 5 découpages aléatoires |
| T-1.6 | Poids réels : parité contre haru_mlx (fp32 et bf16), coût par trame (encodage, décodage) au profiler ; `haru codec roundtrip in.wav out.wav` ⛔ G-1 | encodeur 2e-6 / décodeur 2e-5 fp32 (niveaux amont) ; Vincent écoute l'aller-retour |
