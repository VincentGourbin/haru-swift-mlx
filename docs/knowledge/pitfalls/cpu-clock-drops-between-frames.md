# Piège — l'horloge CPU baisse après ~20 s de petites siestes entre trames (2026-11-17)

**Symptôme** : en vivant (`haru simulate`, `haru talk`), le pas passe de 46 à 57 ms d'un coup après
~150-200 trames (~20-25 s), puis reste haut ; 20 sous-alimentations en 5 min de `haru talk`. Hors
ligne (dos à dos), le pas reste plat (46,5 → 47,4 ms sur 2250 trames).

**Ce que ce n'est pas** (mesuré) : ni la longueur du KV (même saut sans voix, prompt de 268 positions),
ni le cache MLX (même saut à 1 Gio et 4 Gio, cache 0,35 Go), ni l'autoreleasepool unique de la pompe.

**Cause** : la pompe dormait 1 ms entre deux blocs audio hors des 30 dernières ms avant une trame ; le
contrôleur de performance du CPU finit par baisser la fréquence des cœurs P, et la part CPU d'un pas
(construction et envoi de ~300 noyaux) ralentit. Le banc Python `--feed --warm-ms 30` ne le montrait
pas parce qu'il **attend activement en continu** (blocs de 2,7 ms toujours dans la fenêtre de 30 ms) ;
sans chauffe, Python vivant donnait 55,7 ms : le même plateau.

**Correctif** : la pompe sonde en continu pendant une conversation (`warm` = 140 ms > une trame) :
pas plat à **44,1-44,4 ms** sur 450 trames. Coût : un cœur occupé pendant la conversation.

**Règle** : en temps réel à 7,5 Hz, ne pas laisser dormir le fil qui pilote MLX entre deux trames ;
toute mesure vivante dure au moins 60 s (le saut arrive après 20 s). Lien : piège 20 (horloge GPU).
