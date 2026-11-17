# Mémoire d'agent

## Conventions de Vincent (reprises de YuE2 / Sperid)
- Documents en français (plan, fiches, base de connaissances) ; code, commentaires et documentation
  utilisateur (`README.md`, `docs/*.md`) en anglais.
- Un fichier `plan/` par section, `PLAN.md` = sommaire ; fiches `tasks/T-x.y.md` à porte chiffrée ;
  ⛔ GATE = Vincent teste lui-même avant la suite.
- Copier depuis les dépôts frères (`reference/`), jamais de dépendance lourde ; mlx-swift dans la
  plage validée `0.31.6..<0.33.0`.
- Aucun chiffre sans profiler et protocole ; un levier < 5 % est retiré avec sa mesure.
- Rien de publié (push, upload HF, release) sans accord explicite. Dépôt **local** pour l'instant.

## État (2026-11-17)
- T-0.1 validée (squelette, `haru info`) ; poids téléchargés ; T-0.2 en attente d'accord (venv amont).

## Pièges durables
- Haru = une position par trame : régime matrice-vecteur borné par la bande passante, GPU inactif
  ~70 % du temps → mesurer à cadence vivante, juger au p99.
- Référence = haru_mlx (MLX Python), pas PyTorch : parité attendue à l'ulp en fp32 sur la même machine.
