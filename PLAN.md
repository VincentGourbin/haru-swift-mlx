# haru-swift-mlx — Plan d'implémentation

> **Statut : rév. 1 du 2026-11-17 — plan initial**, rédigé après lecture de la référence amont
> (`higurashi-ai/Haru` @ `7fe6b76` : `haru_mlx` — modèle, session, backbone, codec, tête, noyaux
> Metal, poids — et le client Core Audio `swift/Shared/Audio.swift`), de l'en-tête réel du checkpoint
> (`haru info`) et du catalogue de techniques mesurées (`mlx-swift-audit`).
> Objectif : **conversation en temps réel, 100 % Swift, dans le processus**.
> Roadmap : **Jalon 0 (fondations, mesure de référence) → 1 (codec) → 2 (backbone) → 3 (tête,
> session, hors ligne) → 4 (temps réel Core Audio) → 5 (performances, profils) → 6 (app, API,
> iPhone)**. À chaque GATE, Vincent teste lui-même.
> Cible : MacBook Pro M3 Max 96 Go (30 cœurs GPU), macOS 27, Xcode 27.2 (Swift 6.4).
> Poids sous `$HARU_MODELS_DIR` (défaut `~/Library/Caches/models`), jamais de chemin absolu dans le dépôt.

---

## Sommaire — le plan vit dans `plan/`, un fichier par section

**Ce fichier est un sommaire. Ouvrir le fichier de la section demandée, un seul à la fois.**
Les fiches (`tasks/T-x.y.md`) citent ces fichiers par nom.

| Fichier | Section | Mots |
|---|---|---:|
| `plan/00-mode-execution.md` | 0. Mode d'exécution | 158 |
| `plan/01-perimetre.md` | 1. Objectif et périmètre | 290 |
| `plan/02.1-architecture.md` | 2.1 L'architecture, trame par trame (`haru_mlx/models/haru.py`, `session.py`) | 317 |
| `plan/02.2-backbone.md` | 2.2 Backbone Qwen3.5-4B (`haru_mlx/models/qwen3_5.py`, `gated_delta.py`) | 402 |
| `plan/02.3-codec.md` | 2.3 Codec VibeVoice (`haru_mlx/models/vibevoice_codec.py`) | 375 |
| `plan/02.4-tete-meanflow.md` | 2.4 Tête MeanFlow (`haru_mlx/models/meanflow_head.py`) | 229 |
| `plan/02.5-session-prompt.md` | 2.5 Prompt, échantillonnage, détokenisation (`prompt.py`, `sampling.py`, `frames.py`) | 305 |
| `plan/02.6-tenseurs.md` | 2.6 Noms de tenseurs et conversion (`haru_mlx/weights.py`) | 287 |
| `plan/02.7-dtypes.md` | 2.7 Îlots de précision (à respecter pour la parité) | 198 |
| `plan/02.8-temps-reel.md` | 2.8 Temps réel : fils, audio, cadencement (`swift/Shared/Audio.swift`, `worker.py`, `native.py`) | 443 |
| `plan/03-budget.md` | 3. Budget mémoire et temps par trame | 456 |
| `plan/04-jalon0.md` | 4. Jalon 0 — Fondations : squelette, référence mesurée, poids, tokenizer, aléa | 266 |
| `plan/05-jalon1.md` | 5. Jalon 1 — Codec VibeVoice (audio ↔ latents), en flux | 216 |
| `plan/06-jalon2.md` | 6. Jalon 2 — Backbone Qwen3.5 et modèle Haru | 214 |
| `plan/07-jalon3.md` | 7. Jalon 3 — Tête MeanFlow, prompt, session, conversation hors ligne | 159 |
| `plan/08-jalon4.md` | 8. Jalon 4 — Conversation en temps réel (Core Audio dans le processus) | 230 |
| `plan/09-jalon5.md` | 9. Jalon 5 — Performances et profils de référence | 247 |
| `plan/10-jalon6.md` | 10. Jalon 6 — App, API, intégrations, iPhone (étude) | 188 |
| `plan/11-tests.md` | 11. Stratégie de tests et de parité | 295 |
| `plan/12-pieges.md` | 12. Checklist de pièges (contrat de revue — cocher à chaque fiche) | 544 |
| `plan/13-optimisations.md` | 13. Optimisations : ce que nos ports ont mesuré, appliqué au régime « une position par trame » | 576 |
| `plan/14-ecosysteme.md` | 14. Écosystème — brique → dépôt → fichier | 177 |
| `plan/15-risques.md` | 15. Risques | 211 |
| `plan/16-decisions.md` | 16. Décisions actées et questions ouvertes | 178 |

Pour lire le plan d'un bloc (humain seulement) : `Scripts/plan-concat.sh | less`.
