# 1. Objectif et périmètre

## Objectif
Converser **en temps réel** avec Haru-4B sur un Mac Apple Silicon, entièrement en Swift et dans un
seul processus : micro → codec → backbone → tête → codec → haut-parleur, une trame toutes les
133 ms, avec la latence de réponse mesurée sur l'horloge audio.

## Ce que le port apporte par rapport à l'amont
L'amont fournit déjà `haru_mlx` (Python MLX) et `Haru.app`, une app Swift qui **lance un serveur
Python** et lui parle par socket Unix. Ici :
1. Plus de Python à l'exécution : le modèle tourne dans l'app (bibliothèque `HaruCore`), donc
   embarquable (Fluxforge Studio, autres apps) et, à terme, iPhone.
2. Les techniques mesurées de nos ports (YuE2, Qwen3.8, Voxtral, Sperid) appliquées au régime
   « une position par trame » (`plan/13-optimisations.md`), chacune derrière une porte chiffrée.
3. Profils de référence `<bits>bit-fast|lean` au standard de `mlx-swift-audit`.

## Livrables
- `HaruCore` : configuration, poids, backbone, codec, tête MeanFlow, prompt (carte + voix),
  échantillonnage, `DuplexSession` (start / feed / step), `HaruEngine` (fil MLX dédié).
- `HaruAudio` : E/S duplex Core Audio, lecture planifiée, rattrapage WSOLA, mesure de latence.
- `haru` (CLI) : `info`, `download`, `run` (wav → wav), `talk` (conversation au micro), `bench`.
- `haru-studio` (Jalon 6) : app SwiftUI — carte de personnage, voix, deux pistes, transcription.

## Hors périmètre
Backend PyTorch / CUDA / Triton, plugin vLLM, client web, serveur multi-clients (l'API le
permettra, pas le premier livrable), entraînement / fine-tuning, voix fournies (Haru n'en livre
aucune : l'utilisateur apporte un extrait dont il a les droits).

## Cible
MacBook Pro M3 Max 96 Go (14 cœurs CPU, 30 cœurs GPU), macOS 27, Xcode 27.2. iPhone : étude au
Jalon 6 seulement (le modèle 8 bits pèse ≈ 6,3 Go).
