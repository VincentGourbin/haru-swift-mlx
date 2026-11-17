# Journal des fiches

## T-0.1 — Squelette, scripts, configuration, inventaire — 2026-11-17 — validée
- Fait : `Package.swift`, `Scripts/{check-build,check-tests,run-tests,link-references,plan-concat}.sh`, `HaruConfiguration`, `TextConfiguration`, `SafetensorsHeader` et `ModelsDirectory` (copiés de sperid), `WeightInventory`, `haru info`, `ConfigurationTests`
- Bug réel rencontré : aucun (crochet manquant dans la fixture JSON du test, corrigé)
- Validation : `Scripts/check-build.sh` → `BUILD OK` ; `Scripts/check-tests.sh` → `TESTS OK (4 tests, 0 skippés)` ; `haru info` → `total 1022 5270.0 M 10.54 GB`
- Limites connues : pas encore de plan de conversion (T-0.3) ni de SHA-256 épinglé (T-0.6)
- Pas d'agent : 0 · appels d'outils : —

## T-0.3 — Plan de conversion des poids — 2026-11-17 — validée
- Fait : `Loading/WeightPlan.swift` (dispositions, fusions, `check`), `Loading/ParameterShapes.swift`, `haru info --plan`, `WeightPlanTests` + `RealWeightPlanTests` ; défauts amont de `head_dim` et `layer_types` dans `TextConfiguration`
- Bug réel rencontré : aucun (statiques `Regex` non Sendable → `nonisolated(unsafe)`)
- Validation : `Scripts/check-build.sh` → `BUILD OK` ; `Scripts/check-tests.sh` → `TESTS OK (9 tests, 0 skippés)` ; `HARU_MODELS_DIR=… Scripts/check-tests.sh RealWeightPlanTests` → `TESTS OK (1 tests, 0 skippés)` ; `haru info --plan` → `PLAN OK (898 parameters from 1022 tensors)`
- Limites connues : plan seulement, aucune conversion de tenseur (T-2.5)
- Pas d'agent : 0 · appels d'outils : —

## T-0.2 — Mesure de référence — 2026-11-17 — validée
- Fait : `.venv-ref` (installé par Vincent, mlx 0.32.4), 10 passes de `Scripts/reference/baseline.py`, `docs/knowledge/benchmarks/baseline-python-m3max.md`
- Bug réel rencontré : boucle zsh qui passait `--warm-ms 30` comme un seul argument (erreur avalée par `grep`), relancé en commandes explicites
- Validation : 10 JSON, `frames` = 434 chacun ; A/B/B/A vivant bf16 55,9 / 55,5 ms (0,7 %), 8 bits 41,1 / 41,6 ms (1,2 %)
- Limites connues : chauffe bf16 (65,8 ms) en une seule passe, inexpliquée ; pas de voix dans le prompt ; utilisateur = bruit
- Pas d'agent : 0 · appels d'outils : —
