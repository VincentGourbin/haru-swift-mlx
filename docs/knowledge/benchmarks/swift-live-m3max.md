# Port Swift en temps réel sur M3 Max (2026-11-17)

Même machine et mêmes conditions que `baseline-python-m3max.md` (secteur, haute performance, rien
d'autre sur le GPU), mlx-swift 0.32.3, Release, chemins **non compilés** (pas encore de
`compile` par couche : T-5.3).

## Hors ligne (`haru run`, dos à dos, 113 trames, carte en + voix de 180 s)
| Précision | pas p50 | p99 | prompt (1618 positions, voix comprise) |
|---|---:|---:|---:|
| fp32 | 83,3 ms | 117,6 ms | 7,5 s |
| bf16 | 46,5 ms | 47,8 ms | 2,5 s (à chaud) |

Python dos à dos (T-0.2) : bf16 44,5 ms → Swift **+4,5 %**.

## Vivant simulé (`haru simulate`, fichier poussé sur l'horloge murale par blocs de 128 éch. à 48 kHz)
| Précision | pas p50 / p99 / max | fin de trame user → trame modèle prête p50 / p99 | trames en retard |
|---|---|---|---:|
| bf16 (×3) | 46,5 / 47,5–47,7 / 47,7 ms | 50,1 / 51,1–51,3 ms | 0 |
| int8 | 32,9 / 33,8 / 36,5 ms | 36,5 / 37,7 ms | 0 |

Python même chemin (`--feed`, chauffe 30 ms) : bf16 44,7 / 46,3 ; 8 bits 29,3 / 30,1 → Swift
**+4 %** (bf16) et **+12 %** (8 bits). Une première passe bf16 avait montré 3 trames en retard
(max 333 ms) sans pas lent : non reproduit sur 3 passes, à surveiller (endurance T-5.6).

## Core Audio réel (`haru talk`, périphérique virtuel silencieux « Microsoft Teams Audio », 20 s, speak-first)
- Latence de réponse (horloge audio, CAN → CNA) : **médiane 66 ms, p95 68 ms** sur 154 trames.
- Décomposition : entrée 0,7 ms ; moteur 47,7 ms (dont pas 45,3) ; file de sortie 14,3 ms (marge) ; sortie 3,0 ms.
- 1 sous-alimentation (19 ms rattrapés par WSOLA), probablement au démarrage.
- Pas p50 45,3 / p99 50,3 / max 50,6 ms ; 0 échantillon perdu.

## Session de 5 min (`haru talk`, périphérique virtuel, speak-first) — avant / après le correctif de la pompe
| | pas p50 / p99 / max | latence de réponse médiane / p95 | sous-alimentations |
|---|---|---|---:|
| siestes de 1 ms (warm 30 ms) | 56,0 / 64,9 / 107,9 ms | 76 / 80 ms | 20 (384 ms rattrapés) |
| sondage continu (warm 140 ms) | **44,3 / 44,9 / 47,6 ms** | **65 / 65 ms** | 1 (39 ms, au démarrage) |

2255 trames ; empreinte 11 Go ; CPU 61 % d'un cœur (le sondage). Cause : `pitfalls/cpu-clock-drops-between-frames.md`.
Hors ligne sur 2250 trames : Swift 46,5 → 47,4 ms, Python 45,1 → 46,9 ms (pas de dérive notable).
Limite : périphérique virtuel à entrée silencieuse, sans latence matérielle de vrais micro/casque.
