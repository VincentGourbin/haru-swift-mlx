# Décision — aléa : générateur MLX partagé, pas de rejeu (T-0.5, 2026-11-17)

**Question** : les tirages de Swift (mlx-swift) reproduisent-ils ceux de haru_mlx (Python) à graine
égale ? Sinon, les tests de session doivent rejouer le bruit de Python.

**Mesure** (`Scripts/reference/dump_rng.py` → `Tests/HaruTests/Fixtures/rng.json`,
`NoiseSourceTests`) : graines 0 et 1234, 3 pas chacune, dans l'ordre exact de `session.py`
(`split` avant chaque tirage, bruit de tête `normal` puis cast bf16, `codec.sample` qui redécoupe sa
clé, `categorical`) :
- clés (initiale, de tirage, finale) : **identiques** ;
- `normal` fp32 `[1, 64]` (384 valeurs) et bruit de tête bf16 (384) : **identiques** ;
- `categorical` : **identique** ;
- bruit du codec `[1, 3, 64]` (1152 valeurs) : **1 valeur à 1 ulp** (seed 1234, pas 1, indice 167 :
  −0,2677682 contre −0,26776823).

**Cause probable** : mêmes bits aléatoires (clés identiques), transformation bits → loi normale
arrondie différemment sur un élément : mlx-swift 0.32.3 embarque le cœur MLX **v0.32.2**, Python
utilise **0.32.4**. Hypothèse non vérifiée (mlx-swift 0.33.0, hors de notre plage, pourrait aligner).

**Décision** :
1. Pas de rejeu : `NoiseSource` (une clé par session, `split` à chaque tirage) suffit ; les tests de
   tirage exigent l'égalité exacte des clés, du bf16 et de `categorical`, et ≤ 1 ulp sur `normal` fp32.
2. Les tests de session (T-3.4) comparent une conversation à graine fixe **trame par trame avec
   tolérance**, sur un horizon court (50 trames) : 1 ulp de bruit peut, à la longue, faire basculer
   un jeton échantillonné. Au-delà, la porte est « bit-identique à graine fixe » côté Swift (T32).
3. Le point d'injection reste : tout tirage de la session passe par `NoiseSource`. Si T-3.4 montre
   une divergence précoce, on ajoute un rejeu du bruit Python à cet endroit, sans toucher au reste.
4. À chaque montée de mlx-swift : relancer `NoiseSourceTests` (et régénérer la fixture si le
   Python change).
