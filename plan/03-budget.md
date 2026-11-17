# 3. Budget mémoire et temps par trame

> CALCUL = déduit de l'inventaire `haru info` et des chiffres constructeur ; MESURE = observé.
> Toute porte du plan se règle sur la mesure de T-0.2 (référence Python sur cette machine), jamais
> sur ces calculs (piège 25 du catalogue : estimations de gain trop optimistes).

## Octets lus par trame (CALCUL)
Une trame vivante est une suite de produits matrice-vecteur : presque chaque poids est lu une fois.

| Partie | bf16 | 8 bits (amont : backbone + embedding) | 8 bits partout (investigation) |
|---|---:|---:|---:|
| tête de sortie (embedding liée, 248 320 × 2560) | 1,27 Go | 0,68 Go | 0,68 Go |
| backbone | 7,14 Go | 3,79 Go | 3,79 Go |
| tête MeanFlow | 0,73 Go | 0,73 Go | 0,39 Go |
| codec (encodeur + décodeur) | 1,38 Go | 1,38 Go | 0,73 Go |
| **total** | **10,5 Go** | **6,6 Go** | **5,6 Go** |

M3 Max 30 cœurs GPU : 300 Go/s annoncés (à confirmer par micro-banc). Plancher bande passante :
≈ 35 ms (bf16), ≈ 22 ms (8 bits), ≈ 19 ms (8 bits partout). Le budget est 133 ms.

## Chiffres amont (MESURE amont, M1 Max 400 Go/s)
43 ms/trame bf16, 31 ms 8 bits, pic 13,5 / 11,2 Go — soit ≈ 245 et 210 Go/s effectifs. À
efficacité égale sur 300 Go/s : ≈ 57 ms bf16, ≈ 41 ms 8 bits (CALCUL). Le codec seul coûtait
6 ms par encodage ou décodage sur CUDA sans Triton (README amont) : le nombre de dispatchs compte
autant que les octets.

## Croissance avec la conversation (CALCUL)
- KV : 8 couches × 2 × 4 têtes × 256 × 2 o = 32 Ko par position ; 60 min = 27 000 positions =
  0,88 Go, relus à chaque trame en fin de session (+3 ms à 300 Go/s).
- États GDN : 24 × 32 × 128 × 128 × 4 o = 50 Mo, constants (lus et écrits par trame).
- État du codec : < 10 Mo.

## Portes
| Porte | Valeur |
|---|---|
| p99 du pas à cadence vivante, 10 min, 8 bits | ≤ 90 ms (laisse le réveil, la marge et le rattrapage) |
| p99 à 60 min (KV pleine) | ≤ 100 ms |
| sous-alimentations sur 10 min, périphériques filaires | 0 |
| latence de réponse médiane (filaire) | ≤ amont mesuré + 10 ms |
| pic mémoire bf16 / 8 bits | ≤ 14 / 11,5 Go (amont : 13,5 / 11,2) |
| démarrage avec préfixe préparé | ≤ 1 trame |
