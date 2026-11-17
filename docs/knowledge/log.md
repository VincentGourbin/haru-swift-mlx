# Journal

## 2026-11-17 — création du projet
- Lecture de `higurashi-ai/Haru` @ `7fe6b76` : haru_mlx (modèle, session, backbone Qwen3.5, codec VibeVoice, tête MeanFlow, noyaux Metal GDN et mélangeur depthwise, plan de poids), client Core Audio `swift/Shared/Audio.swift`, cadencement `worker.py` / `backends.py` / `native.py`.
- Poids téléchargés (`hf download`, 10,54 Go) sous `$HARU_MODELS_DIR/higurashi-ai/Haru-4B`.
- `haru info` (MESURE de l'en-tête) : 1022 tenseurs, 5 270 M paramètres ; embedding 1,27 Go, backbone 7,14 Go, tête 0,73 Go, codec 2 × 0,69 Go, projecteurs 0,03 Go. Soit ≈ 10,5 Go lus par trame en bf16 (CALCUL, `plan/03-budget.md`).
- Machine : M3 Max 14 CPU / 30 GPU, 96 Go — bande passante annoncée 300 Go/s contre 400 pour le M1 Max des chiffres amont (43 / 31 ms par trame) : le bf16 risque d'être plus serré ici (CALCUL ≈ 57 ms).
- Le GDN de mlx-swift-lm (vendored dans qwen38) diffère du noyau de haru_mlx (somme de Kahan) : piège n° 3.
- Mesure de référence Python reportée (installation de haru_mlx refusée par le mode auto, en attente de l'accord de Vincent : `tasks/ASK.md`).
- SwiftPM a résolu mlx-swift **0.32.3** (plage 0.31.6..<0.33.0), swift-transformers 1.4.0, swift-mlx-profiler 1.5.1.
- T-0.3 : plan de conversion porté, vérifié sur l'en-tête réel (898 paramètres, 0 écart) : les faits de `plan/02.6` sont confirmés par le checkpoint, pas seulement par lecture.
- Le générateur tiny amont (`make_tiny_checkpoint.py`) exige torch et le paquet PyTorch `haru` : T-1.1 écrira son propre générateur numpy.

## 2026-11-17 — T-0.2, mesure de référence
- haru_mlx sur M3 Max : bf16 vivant p50 55,7 ms (dos à dos 44,5), 8 bits 41,3 (30,6) ; `feed` + chauffe 44,7 / 29,3 ms ; pic MLX 11,2 / 6,8 Go. Le piège 20 est confirmé (+25 / +35 % en vivant).
- G-0 levée : bf16 par défaut (p99 vivant ≈ 63 ms, porte 90 ms), dépôt public, voix de test locale hors dépôt, app autonome.
