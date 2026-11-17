# Piège — MLX Swift et MLX Python ne compilent pas leurs noyaux Metal pareil (2026-11-17)

**Constat** : parité du backbone en bf16 à 1e-2 – 5e-2 au lieu de 0 (le codec, lui, est bit-exact).
L'écart naît dès la première opération de la couche 0 (`input_layernorm`, 5,9e-6 en bf16) puis
s'accumule sur 32 couches. En fp32, tout est à ~1e-5 et le top-1 est identique.

**Cause, isolée** : mêmes poids (écart 0), même entrée, même appel
`mx.fast.rms_norm(x.float32, w, 1e-6)` : la sortie fp32 diffère de **3,5e-8** entre Python (mlx
0.32.4, et aussi 0.32.2) et mlx-swift 0.32.3 (cœur v0.32.2). Ce n'est donc pas la version : Python
livre une metallib précompilée, mlx-swift compile les sources Metal avec Xcode (drapeaux et
compilateur différents). Un écart de 1e-7 suffit à faire basculer quelques valeurs au bf16 voisin.

**Règle** : en bf16, la parité Swift ↔ Python se juge par le même top-1 et des écarts dans le bruit
bf16↔fp32 de l'amont (6e-2 – 9e-2), jamais au bit ; la parité au bit se vérifie en fp32 (≤ 5e-5) et,
côté Swift seul, à graine fixe (T32). Les tests de session en bf16 ne peuvent pas comparer une
conversation entière au jeton près : comparer en fp32, ou les premières trames seulement.
