# 13. Optimisations : ce que nos ports ont mesuré, appliqué au régime « une position par trame »

Source : catalogue `mlx-swift-audit/references/techniques.md` (YuE2 = Y, Qwen3.8 = Q, Voxtral = V).
Régime ici : décodage **borné par la bande passante** (10,5 Go lus par trame en bf16), une trame
toutes les 133 ms, GPU majoritairement inactif entre deux trames. Rien n'est acquis : chaque ligne
est une fiche du Jalon 5 avec porte (p99 −5 % à cadence vivante ou retrait).

## Déjà dans la conception amont (à porter, puis à vérifier)
| Levier | Origine | Note |
|---|---|---|
| Projections fusionnées (in_proj GDN, qkv, gate_up, ada) | amont | moins de dispatchs, produits plus grands |
| Une synchronisation par trame | V T24 | `_emit` ne fait qu'un `eval` |
| Avant de l'encodeur au fil du micro (`feed`) | amont | retire ≈ 7/8 de l'encodeur du chemin critique |
| 8 bits backbone + embedding | amont, Y/Q T13 | 43 → 31 ms sur M1 Max |
| Préfixe préparé (carte + voix) | amont PREPARE, V T25 | ici : clone d'état, pas de serveur |
| Limite de cache 1 Gio + `clearCache` après prompt + limite câblée | amont, Y/Q T1-T3 | |
| Chauffe CPU avant l'échéance | amont `worker.py` | réveil de cœur évité |

## Candidats (Jalon 5, par gain attendu)
| # | Levier | Origine | Idée | Risque |
|---|---|---|---|---|
| O1 | 8 bits de la tête MeanFlow et des étages ≥ 1024 canaux du codec | T13 (mixte par voie) | −1 Go/trame (≈ −15 %) | qualité audio : juge ASR + écoute (R11 : étage sensible) |
| O2 | Tête de sortie (embedding liée) plus basse précision ou restreinte | T12 | 1,27 Go/trame en bf16 ; PAD/EOP restent en fp32 | jetons qui changent ; pas de restriction de vocabulaire sans mesure |
| O3 | `asyncEval` : trame t+1 lancée dès que la trame user est prête, lecture CPU (jeton, DRY, texte) et décodage audio en recouvrement | Q T15, V | sortir les syncs du chemin critique | exactitude du DRY (pénalité à t+1 dépend du jeton t) |
| O4 | Compilé vs non compilé, noyaux vs repli | Q R1-R3, R9 | les rejets Q sont en régime débit ; ici 1 position | A/B obligatoire |
| O5 | Pack préquantifié (chargement direct) | Y, piège 12 du catalogue | démarrage, pic de chargement | nom du pack = profil |
| O6 | Maintien d'horloge GPU (charge de garde ou cadence) | nouveau | p99 si la DVFS coûte | énergie, chaleur : mesurer d'abord (piège 20) |
| O7 | KV quantifié 8 bits | Y/Q T10 | 0,88 → 0,45 Go à 60 min, −1,5 ms en fin de session | parité ; gain faible : seulement si T-5.6 le montre |
| O8 | Lecture des poids `pread` concurrents | T22 | chargement à froid | — |

## Rejetés d'avance (avec raison)
- Décodage spéculatif / MTP (Q : 0,94× ; ici une seule position par trame et l'entrée dépend de
  l'utilisateur).
- CFG en deux passes (la guidance est une entrée de la tête : déjà une passe).
- Core ML / ANE pour le codec (Y R14 : 5-8× plus lent ; V R25 non prouvé) — à rouvrir seulement
  pour l'iPhone.
