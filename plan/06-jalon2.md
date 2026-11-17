# 6. Jalon 2 — Backbone Qwen3.5 et modèle Haru

| Fiche | Contenu | Porte |
|---|---|---|
| T-2.1 | `RMSNorm` (gain fp32 1+w), `GatedNorm`, `MLP` (`gate_up`), `AudioProjector` | `PARITY OK layers` |
| T-2.2 | Règle delta : noyau Metal **de haru_mlx** (pas celui de mlx-swift-lm, qui ajoute une somme de Kahan) + repli tableaux | noyau = repli ≤ 1e-5 sur 64 pas ; parité haru_mlx |
| T-2.3 | `GatedDeltaNet` (état conv + mémoire explicites), `Attention` (q/gate par tête, normes avant RoPE, RoPE 64 non entrelacée), `KVCache` par blocs de 256 | `PARITY OK gdn`, `PARITY OK attention` |
| T-2.4 | `DecoderLayer`, `Qwen35TextModel` : chemin multi-positions et chemin une position compilé (GDN : une fonction ; attention : avant/après cache) | une position compilée = non compilée ≤ 1e-6 |
| T-2.5 | Chargeur : `WeightPlan` (T-0.3) appliqué tenseur par tenseur, `eval` par partie ; `HaruModel` : `embed` (3 flux + masques), `logits`, `logitsFP32` | tiny : `PARITY OK haru` (prefill + 6 pas, comme `dump_reference.py`) |
| T-2.6 | Poids réels : prefill d'une carte + 24 trames, puis 6 pas ; comparaison à haru_mlx ; pic de chargement | fp32 : 1e-5 sur les états cachés, même top-1 ; bf16 : dans l'écart bf16↔fp32 de haru_mlx |
