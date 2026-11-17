# Piège n° 20 — à 7,5 Hz, le GPU s'endort entre deux trames (mesuré)

**Hypothèse** (2026-11-17) : une trame demande ≈ 30-60 ms de GPU toutes les 133 ms. Entre deux
trames, le GPU est inactif et sa fréquence peut baisser ; le pas mesuré dos à dos
(`run_inference`, 43 / 31 ms amont) peut donc sous-estimer le pas vivant. L'amont garde déjà le
CPU éveillé avant l'échéance (`worker.py`, `warm` 30 ms), pas le GPU.

**Mesure prévue** : T-0.2 compare `--cadence live` et `--cadence burst` à précision égale.
**Règle d'ici là** : aucune porte temps réel sur un banc dos à dos.

**Mesure (T-0.2, 2026-11-17)** : bf16 vivant 55,7 ms contre 44,5 dos à dos (+25 %) ; 8 bits 41,3
contre 30,6 (+35 %), A/B/B/A à 1 % près. La chauffe CPU de 30 ms ramène le 8 bits à 37,0 ms, mais
une passe bf16 avec chauffe a donné 65,8 ms (inexpliqué, à refaire). La part CPU (réveil) et la
part GPU (horloge) restent à séparer au profiler dans le port Swift (levier O6).
