# 4. Jalon 0 — Fondations : squelette, référence mesurée, poids, tokenizer, aléa

But : tout ce dont les jalons de modèle ont besoin, et une **mesure de référence** sur cette
machine avant d'écrire une seule couche.

| Fiche | Contenu | Porte |
|---|---|---|
| T-0.1 | Squelette : `Package.swift`, scripts, `HaruConfiguration`, `SafetensorsHeader`, `WeightInventory`, `haru info` | `BUILD OK`, `TESTS OK (4 tests)` — **validée 2026-11-17** |
| T-0.2 | Environnement Python de référence `.venv-ref` (`pip install -e reference/haru/haru_mlx`), `Scripts/reference/baseline.py` : `run_inference` sur un enregistrement de 60 s, bf16 puis `--quantize 8`, distribution `frame.seconds` (p50/p95/p99/max) **à cadence vivante** (une trame / 133 ms) et dos à dos ; pic mémoire ⛔ G-0 | fichier `docs/knowledge/benchmarks/baseline-python-m3max.md` rempli |
| T-0.3 | `WeightPlan` : port de `weights.py` (plan de conversion par noms/formes, `parameterShapes(config)`, `checkPlan`) ; test contre l'en-tête réel (1022 tenseurs → paramètres attendus, 0 écart) | `TESTS OK`, `haru info --plan` affiche `PLAN OK` — **validée 2026-11-17** (898 paramètres, 0 écart) |
| T-0.4 | Tokenizer (`swift-transformers`, `tokenizer.json`) + rendu de carte (`card_templates.json` copié dans les ressources) + `systemPromptIds` ; ids comparés à Python sur les 9 cartes d'exemple | ids identiques |
| T-0.5 | `NoiseSource` (clé MLX, `split`) ; fixture : mêmes clés → mêmes `normal` / `categorical` en Python et en Swift (sinon : bruit rejoué depuis la fixture dans tous les tests de parité) | décision consignée `decisions/rng.md` |
| T-0.6 | `haru download` : copie de `HubDownloader` de sperid (Range, SHA-256 épinglé) | SHA-256 du checkpoint local vérifié |

⛔ G-0 : Vincent valide la mesure de référence et choisit la précision par défaut (bf16 / 8 bits).
