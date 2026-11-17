# Mesure de référence — haru_mlx amont sur M3 Max (T-0.2, 2026-11-17)

**Conditions** : MacBook Pro M3 Max 14 CPU / 30 GPU, 96 Go, secteur, profil haute performance,
macOS 27, Xcode 27.2 ; Python 3.12.5, **mlx 0.32.4**, haru_mlx @ `7fe6b76` (haru-speech-mlx 0.1.0)
en editable dans `.venv-ref` ; checkpoint `higurashi-ai/Haru-4B` (bf16, 10,54 Go) dans le cache
des pages (chargement 0,6-0,9 s). Aucune autre charge GPU (Ollama sans modèle chargé). Script :
`Scripts/reference/baseline.py`, 60 s = 450 trames dont 16 de chauffe écartées (434 mesurées),
côté utilisateur = bruit faible, prompt « You are Haru. » sans voix, graine 0, limite de cache
1 Gio, limite câblée = working set recommandé. 20 s de pause entre deux passes. JSON bruts dans
`.local-runs/bench.noindex/` (non committés).

`frame.seconds` = temps de `step()` (encodage de la trame utilisateur, position du backbone, tête,
décodage, `eval`, lecture du jeton). Avec `--feed`, l'avant de l'encodeur a déjà tourné pendant la
trame : il n'est **pas** compté.

| Passe (ordre d'exécution) | Précision | Cadence | p50 | p95 | p99 | max | > 133 ms | pic MLX |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | bf16 | vivante | 55,9 | 58,6 | 63,6 | 150,3 | 1 | 11,22 Go |
| 2 | 8 bits | vivante | 41,1 | 46,0 | 47,1 | 50,1 | 0 | 6,78 Go |
| 3 | 8 bits | vivante | 41,6 | 46,6 | 48,2 | 49,5 | 0 | 6,80 Go |
| 4 | bf16 | vivante | 55,5 | 60,0 | 61,9 | 63,9 | 0 | 11,22 Go |
| 5 | bf16 | dos à dos | 44,5 | 45,2 | 45,7 | 46,0 | 0 | 11,22 Go |
| 6 | 8 bits | dos à dos | 30,6 | 31,0 | 31,2 | 31,5 | 0 | 6,78 Go |
| 7 | 8 bits | vivante + chauffe 30 ms | 37,0 | 39,0 | 40,2 | 41,0 | 0 | 6,78 Go |
| 8 | 8 bits | vivante + chauffe + `--feed` | 29,3 | 29,7 | 30,1 | 30,8 | 0 | 6,80 Go |
| 9 | bf16 | vivante + chauffe 30 ms | 65,8 | 66,5 | 66,8 | 68,9 | 0 | 11,22 Go |
| 10 | bf16 | vivante + chauffe + `--feed` | 44,7 | 45,6 | 46,3 | 48,0 | 0 | 11,22 Go |

(ms ; passes 9-10 lancées après la série, 2 min plus tard.)

## Lecture
- **Reproductibilité** : A/B/B/A vivant, bf16 55,9 / 55,5 (0,7 %), 8 bits 41,1 / 41,6 (1,2 %) ✔.
- **Piège 20 confirmé** : à cadence vivante, le pas coûte **+25 % (bf16)** et **+35 % (8 bits)**
  par rapport au dos à dos. Les chiffres dos à dos sont ceux que publie l'amont (`run_inference`).
- **Chauffe CPU** (amont `worker.py`) : 8 bits 41,3 → 37,0 ms (−10 %) ; bf16 55,7 → **65,8 ms
  (+18 %)**, inexpliqué. Une seule passe, après 10 min de charge : à refaire en A/B/B/A avant toute
  conclusion (thermique ? sonde Python qui occupe le cœur du fil MLX ?).
- **`--feed`** retire l'avant de l'encodeur du pas : 8 bits 37,0 → 29,3 ms, bf16 → 44,7 ms. C'est le
  chemin que le port Swift suivra ; c'est lui la référence des portes du Jalon 4.
- **Débit effectif dos à dos** (CALCUL depuis l'inventaire) : bf16 10,5 Go / 44,5 ms ≈ 236 Go/s ;
  8 bits 6,6 Go / 30,6 ms ≈ 216 Go/s. Mieux que l'estimation de `plan/03` (57 ms bf16) : le M3 Max
  tient ≈ 79 % des 300 Go/s annoncés.
- **Mémoire** : pic MLX 11,2 Go (bf16) / 6,8 Go (8 bits), sous les chiffres amont M1 Max
  (13,5 / 11,2 Go, empreinte processus).
- **Un dépassement** : passe 1 (bf16), une trame à 150 ms. Première passe de la série, isolée.
  À surveiller dans l'endurance (T-5.6).
- **Budget** : bf16 vivant p99 62-64 ms pour 133 ms de budget, porte 90 ms ✔ : la décision D8
  (bf16 par défaut) tient avec ≈ 30 % de marge sur la porte.

## Références des portes Swift (Jalon 4, même machine, même protocole)
- bf16, vivante, `feed` + chauffe : p50 44,7 ms, p99 46,3 ms.
- 8 bits, vivante, `feed` + chauffe : p50 29,3 ms, p99 30,1 ms.
Le port Swift doit faire au moins aussi bien (tolérance 5 %).
