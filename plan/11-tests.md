# 11. Stratégie de tests et de parité

## Trois niveaux
1. **Sans poids** (toujours vert, CI possible) : configuration, plan de conversion sur en-tête
   synthétique, DRY, top-k/top-p, détokeniseur, prompt (positions et masques), WSOLA, Player
   (horloge simulée), rééchantillonneurs.
2. **Tiny** (`parity/*.safetensors`, générés par script Python seulement) : checkpoint aléatoire
   graine fixe aux dimensions réduites (`tests/parity/tiny_config.json` amont : toutes les sortes
   de couches), entrées et sorties dumpées par **haru_mlx** ; tolérance fp32 1e-5.
3. **Réel** (`Real*Tests.swift`, `HARU_MODELS_DIR`) : mêmes entrées que `dump_reference.py`
   (carte « You are Haru. », 24 positions de référence, 6 pas ; clip de 5 trames + 1234 éch.) ;
   cibles fp32 = niveaux amont (états 9e-6 prompt / 6e-6 pas, tête 2e-6, codec 2e-6 / 2e-5).

## Référence = haru_mlx, même machine
Les dumps sont faits par `Scripts/reference/*.py` avec `.venv-ref` (haru_mlx en editable). Comme
les deux côtés sont MLX sur le même GPU, l'écart attendu en fp32 est de l'ordre de l'ulp ; un
écart de 1e-3 signale une erreur de port, pas de l'arrondi.

## Aléa
Tranché par T-0.5 (`docs/knowledge/decisions/rng.md`) : clés, bf16 et `categorical` identiques à
Python, `normal` fp32 à ≤ 1 ulp. Les tests de session comparent 50 trames à graine fixe avec
tolérance ; au-delà, « bit-identique à graine fixe » côté Swift (T32). Pas de rejeu tant que T-3.4
ne montre pas de divergence précoce.

## Temps réel
- Banc `haru bench --cadence live|burst --minutes N` : pas par trame, p50/p95/p99/max, pic
  mémoire, horloge GPU observée ; sorties sous `.local-runs/bench.noindex/`.
- Entrée virtuelle : un WAV joué au rythme de l'horloge audio à la place du micro (reproductible),
  puis seulement la conversation vraie (GATE).
- Qualité : transcription ASR de la voix du modèle (juge commun T33, Voxtral local) sur un jeu
  fixe de 5 conversations, pour juger les quantifications.
