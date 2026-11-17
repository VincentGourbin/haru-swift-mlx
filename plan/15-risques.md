# 15. Risques

| Risque | Effet | Parade |
|---|---|---|
| Bande passante M3 Max 30 cœurs (300 Go/s) < M1 Max (400) | pas bf16 ≈ 57 ms au lieu de 43 | 8 bits par défaut si T-0.2 le confirme ; O1/O2 |
| DVFS à 7,5 Hz | p99 très au-dessus du banc | mesure à cadence vivante (piège 20), O6 |
| Aléa MLX Swift ≠ Python | pas de comparaison de conversation complète | rejeu du bruit (T-0.5) |
| Noyaux Metal custom en Swift (API `MLXFast.metalKernel`) | écarts, plantage | repli tableaux gardé et testé contre le noyau |
| Fonctions compilées à état explicite en Swift | retrace à chaque trame | A/B (T-5.3), formes fixes |
| Fil audio bloqué par MLX | craquements | piège 22, tampons circulaires, test d'endurance |
| 4 bits divergent (amont) | pas de modèle léger pour iPhone | investigation, pas de promesse |
| Modèle de base, pas assistant | réponses non factuelles | c'est le modèle publié : documenter (README) |
| Clonage de voix | usage abusif | aucune voix livrée, avertissement dans l'app (comme l'amont) |
| Changement amont (0.1.1 → …) | parité qui casse | commit épinglé, montée explicite |
