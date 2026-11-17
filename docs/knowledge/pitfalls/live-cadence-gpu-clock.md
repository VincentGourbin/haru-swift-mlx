# Piège n° 20 — à 7,5 Hz, le GPU s'endort entre deux trames (hypothèse, à mesurer)

**Hypothèse** (2026-11-17) : une trame demande ≈ 30-60 ms de GPU toutes les 133 ms. Entre deux
trames, le GPU est inactif et sa fréquence peut baisser ; le pas mesuré dos à dos
(`run_inference`, 43 / 31 ms amont) peut donc sous-estimer le pas vivant. L'amont garde déjà le
CPU éveillé avant l'échéance (`worker.py`, `warm` 30 ms), pas le GPU.

**Mesure prévue** : T-0.2 compare `--cadence live` et `--cadence burst` à précision égale.
**Règle d'ici là** : aucune porte temps réel sur un banc dos à dos.
