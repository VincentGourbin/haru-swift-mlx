# 14. Écosystème — brique → dépôt → fichier

`reference/` est créé par `Scripts/link-references.sh` (gitignoré).

| Brique | Source | Fichier |
|---|---|---|
| Référence numérique | `reference/haru` @ `7fe6b76` (higurashi-ai/Haru, Apache-2.0) | `haru_mlx/haru_mlx/**` |
| Fixtures tiny | idem | `haru_mlx/tests/parity/{make_tiny_checkpoint.py,tiny_config.json,dump_reference.py}` |
| Client Core Audio, Player, WSOLA | idem | `haru_mlx/swift/Shared/Audio.swift` |
| Cadencement côté modèle | idem | `haru_mlx/haru_mlx/{worker.py,backends.py,native.py}` |
| Cartes, gabarits | idem | `haru_mlx/haru_mlx/assets/{card_templates.json,cards/*.json}` |
| En-tête safetensors, dossier modèles, téléchargement | `reference/sperid` | `Sources/SperidCore/Loading/*` (copiés) |
| Scripts check-*, run-tests, AGENTS | `reference/sperid` | `Scripts/*`, `AGENTS.md` |
| `MLXWorker` Swift (fil MLX unique) | `reference/sperid` | `Sources/IdolCore/API/MLXThread.swift` + `docs/knowledge/pitfalls/mlx-cross-thread-app-crash.md` |
| GDN Swift (lecture seulement : différences, piège 3) | `reference/qwen38/Vendor/mlx-swift-lm` | `Libraries/MLXLMCommon/GatedDelta.swift`, `Libraries/MLXLLM/Models/Qwen35.swift` |
| Juge ASR, enregistrement, TTS/STT temps réel | `reference/voxtral` | Realtime, `AVAudioConverter` (piège 24) |
| Profils, banc, base de connaissances | `reference/yue2`, `reference/qwen38` | `*ReferenceProfile`, `docs/knowledge` |
| Profiler | `reference/profiler` (swift-mlx-profiler ≥ 1.5.1) | dépendance SwiftPM |

Dépendances SwiftPM : mlx-swift `0.31.6..<0.33.0`, swift-argument-parser, swift-mlx-profiler,
swift-transformers (Tokenizers seulement). Pas de mlx-swift-lm (copie, pas dépendance).
