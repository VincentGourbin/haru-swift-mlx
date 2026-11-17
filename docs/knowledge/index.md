---
okf_version: "0.1"
---

# Base de connaissances — haru-swift-mlx

Journal d'ingénierie du portage de Haru-4B (bundle OKF). `log.md` porte l'historique horodaté ; les
sous-répertoires accumulent les conclusions durables.

## Benchmarks
- (T-0.2) `benchmarks/baseline-python-m3max.md` — à créer : haru_mlx amont sur ce M3 Max, bf16 / 8 bits, cadence vivante / dos à dos.

## Decisions
- [Organisation du port](decisions/port-layout.md) — 100 % Swift dans le processus, référence haru_mlx sur la même machine, pas de mlx-swift-lm, client Core Audio amont copié.

## Pitfalls
Numérotés comme `plan/12-pieges.md`.
- [Deux noyaux GDN qui ne donnent pas le même résultat](pitfalls/gdn-kernel-kahan.md) — celui de mlx-swift-lm fait une somme de Kahan, celui de haru_mlx non (n° 3).
- [Le GPU s'endort entre deux trames](pitfalls/live-cadence-gpu-clock.md) — à 7,5 Hz, un banc dos à dos ment (n° 20, hypothèse à mesurer en T-0.2).

## Investigations
Rien pour l'instant.

## Transmissions et mémoire
- [Mémoire d'agent](agent-memory.md) — conventions de Vincent, état du projet : à lire en début de session.
