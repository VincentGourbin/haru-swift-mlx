# Piège n° 3 — deux noyaux Gated DeltaNet, deux résultats

**Constat** (lecture de code, 2026-11-17) : `mlx-swift-lm/Libraries/MLXLMCommon/GatedDelta.swift`
calcule `kv_mem = Σ state·k` avec une somme de Kahan sous `#pragma clang fp reassociate(off)` /
`contract(off)` ; le noyau de `haru_mlx/models/gated_delta.py` fait une somme simple. Même
algorithme, arrondis différents, et l'état se compose sur toute la conversation.

**Règle** : copier le noyau de haru_mlx (référence de parité). La variante Kahan peut être essayée
au Jalon 5 comme levier de précision, mesurée, jamais substituée en silence.
