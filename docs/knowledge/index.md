---
okf_version: "0.1"
---

# Base de connaissances — haru-swift-mlx

Journal d'ingénierie du portage de Haru-4B (bundle OKF). `log.md` porte l'historique horodaté ; les
sous-répertoires accumulent les conclusions durables.

## Benchmarks
- [Référence haru_mlx sur M3 Max](benchmarks/baseline-python-m3max.md) — bf16 vivant p50 55,7 / p99 63 ms, dos à dos 44,5 ; 8 bits 41,3 / 30,6 ; `feed` + chauffe : bf16 44,7, 8 bits 29,3 ms ; pic 11,2 / 6,8 Go ; ≈ 79 % des 300 Go/s.

## Decisions
- [Organisation du port](decisions/port-layout.md) — 100 % Swift dans le processus, référence haru_mlx sur la même machine, pas de mlx-swift-lm, client Core Audio amont copié.

## Pitfalls
Numérotés comme `plan/12-pieges.md`.
- [Deux noyaux GDN qui ne donnent pas le même résultat](pitfalls/gdn-kernel-kahan.md) — celui de mlx-swift-lm fait une somme de Kahan, celui de haru_mlx non (n° 3).
- [Le GPU s'endort entre deux trames](pitfalls/live-cadence-gpu-clock.md) — à 7,5 Hz, le pas vivant coûte +25 % (bf16) / +35 % (8 bits) de plus que dos à dos (n° 20, mesuré T-0.2).

## Investigations
Rien pour l'instant.

## Transmissions et mémoire
- [Mémoire d'agent](agent-memory.md) — conventions de Vincent, état du projet : à lire en début de session.
