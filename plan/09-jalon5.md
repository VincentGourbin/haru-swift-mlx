# 9. Jalon 5 — Performances et profils de référence

Chaque levier de `plan/13-optimisations.md` est une fiche à porte chiffrée : A/B/B/A à cadence
vivante, gain ≥ 5 % du p99 sinon retiré avec sa mesure.

| Fiche | Contenu | Porte |
|---|---|---|
| T-5.1 | Quantification 8 bits amont (couches du backbone une par une, puis embedding ; re-compilation) + export d'un pack préquantifié (chargement sans conversion) | états cachés 8 bits vs fp32 dans l'écart bf16 ; p99 mesuré |
| T-5.2 | Profils `bf16-fast`, `8bit-fast`, `8bit-lean` (cache MLX, limite câblée, chauffe, marge) figés et mesurés | tableau de `BENCHMARKS.md` rempli |
| T-5.3 | A/B des chemins : compilé vs non compilé par couche, noyaux Metal vs repli (les rejets R1-R3/R9 de Qwen3.8 viennent d'un régime de débit, pas d'une position par trame) | garder le plus rapide, consigner l'autre |
| T-5.4 | Recouvrement : décodage codec de la trame t et lecture CPU du jeton hors du chemin critique (`asyncEval`, DRY en décalé si exact) | p99 −5 % ou retrait |
| T-5.5 | Investigation : 8 bits de la tête MeanFlow et des étages profonds du codec (−1 Go/trame) ; tête de sortie 4 bits ; 4 bits mixte backbone | juge ASR commun (T33) + écoute ; pas de profil sans parité |
| T-5.6 | Endurance 60 min : p99 en fin de session (KV pleine), pic mémoire, dérive thermique | portes 60 min de `plan/03-budget.md` |
