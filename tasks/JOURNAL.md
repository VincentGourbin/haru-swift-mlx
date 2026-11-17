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

## T-0.4 — Tokenizer, cartes, ids du system prompt — 2026-11-17 — validée
- Fait : `Prompt/CharacterCard.swift` (`CharacterCard`, `CardTemplate.render`, `CardLibrary`), `Prompt/HaruTokenizer.swift` (swift-transformers 1.4.0), ressources `Resources/Cards/` (amont), `Scripts/reference/dump_cards.py` → `Tests/HaruTests/Fixtures/cards.json`, `CardTests` + `RealTokenizerTests`, `haru card`
- Bug réel rencontré : aucun
- Validation : `Scripts/check-build.sh` → `BUILD OK` ; `Scripts/check-tests.sh` → `TESTS OK (14 tests, 0 skippés)` ; `HARU_MODELS_DIR=… Scripts/check-tests.sh RealTokenizerTests` → `TESTS OK (2 tests, 0 skippés)` (9 cartes + 8 chaînes : ids identiques, aller-retour decode)
- Limites connues : le détokeniseur de flux (PAD/EOP → espace, U+FFFD retenu) est en T-3.2
- Pas d'agent : 0 · appels d'outils : —

## T-0.5 — NoiseSource et aléa Python ↔ Swift — 2026-11-17 — validée
- Fait : `Sampling/NoiseSource.swift`, `Scripts/reference/dump_rng.py` → `Tests/HaruTests/Fixtures/rng.json`, `NoiseSourceTests`, `docs/knowledge/decisions/rng.md`
- Bug réel rencontré : bruit du codec à 1 ulp sur 1 valeur sur 1152 (cœur MLX v0.32.2 dans mlx-swift 0.32.3 contre 0.32.4 en Python, hypothèse) ; règle ≤ 1 ulp sur `normal` fp32
- Validation : `Scripts/check-tests.sh NoiseSourceTests` → `TESTS OK (2 tests, 0 skippés)` ; `Scripts/check-tests.sh` → `TESTS OK (16 tests, 0 skippés)`
- Limites connues : la divergence d'une conversation à graine fixe sur un long horizon reste à mesurer (T-3.4)
- Pas d'agent : 0 · appels d'outils : —

## T-0.6 — Téléchargement épinglé — 2026-11-17 — validée
- Fait : `Loading/{HubFile,HubDownloader,ChunkedDownloadDelegate,HubToken}.swift` (copiés de sperid, cas « gated » retiré), `Loading/HaruWeights.swift` (révision `8d6e83bc`, 4 fichiers, tailles et SHA-256), `haru download [--verify]`, `HaruWeightsTests`, `HubDownloaderTests` (serveur bouchon), `RealWeightsTests`
- Bug réel rencontré : aucun
- Validation : `Scripts/check-tests.sh` → `TESTS OK (23 tests, 0 skippés)` ; `HARU_MODELS_DIR=… Scripts/check-tests.sh RealWeightsTests` → `TESTS OK (1 tests, 0 skippés)` ; `haru download --verify` → `DOWNLOAD OK (4 files present, SHA-256 verified, revision 8d6e83bc)`
- Limites connues : le chemin réseau réel n'est pas rejoué (les poids étaient déjà là : `hf download`)
- Pas d'agent : 0 · appels d'outils : —

## Jalons 1-4 + app de test — 2026-11-17 — validés (portes automatiques, D12/D13)
- Fait : codec (`Models/Codec/*`), backbone (`Models/Backbone/*`), tête (`Models/Head/*`), `HaruModel`, `WeightStore`, `Dense`, échantillonnage (`Sampling/TextSampling.swift`), `AudioFile`, `DuplexSession`, `MLXWorker`, `HaruEngine`, `LiveConversation`, `HaruAudio` (client Core Audio amont), `haru run / talk / devices / simulate`, app `HaruStudio` + `Scripts/package-app.sh`, dumps `Scripts/reference/{dump_parity,dump_session}.py`
- Bugs réels rencontrés : bf16 non bit-exact (noyaux Metal compilés différemment, piège consigné) ; `self` capturé avant init ; voix chargée deux fois (`@Observable` déclenche `didSet` dans `init`) ; voix illisible dans l'app depuis `~/Downloads` (TCC) ; boucle zsh sans découpage d'arguments
- Validation : `RealCodecParityTests` (bf16 0, fp32 ≤ 1,4e-5) ; `RealModelParityTests` (fp32 ≤ 1,4e-5, top-1 identique) ; session fp32 : 34 trames identiques à Python ; `haru simulate` bf16 3 passes, 0 trame en retard ; `haru talk` (périphérique virtuel) latence médiane 66 ms ; app « Ready » en 2,5 s ; `Scripts/check-tests.sh` → `TESTS OK (29 tests)`
- Limites connues : pas de test au vrai micro ni du bouton Start de l'app ni de « Save conversation… » ; chemins non compilés ; pas de fixtures tiny ; 1 sous-alimentation au démarrage de `haru talk`
- Complément (même jour) : session de 5 min → dérive du pas (siestes de la pompe, piège consigné) corrigée par sondage continu : 44,3 / 44,9 ms, 1 sous-alimentation ; courses Start/End corrigées
- Pas d'agent : 0 · appels d'outils : —
