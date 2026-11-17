# Décision — organisation du port (2026-11-17)

**Contexte** : l'amont fournit haru_mlx (Python MLX) et une app Swift qui lance un serveur Python
(socket Unix). L'objectif de Vincent est une conversation temps réel « avec nos connaissances et
optimisations », comme pour sperid.

**Décision** :
1. 100 % Swift dans le processus (`HaruCore` + `HaruAudio`), un seul fil MLX ; pas de socket.
2. Référence numérique = haru_mlx sur la même machine ; noms et paramètres fusionnés repris à
   l'identique pour une parité tenseur par tenseur.
3. Pas de dépendance à mlx-swift-lm : le backbone est réécrit à partir de haru_mlx (disposition
   fusionnée, chemin une position compilé), ce que mlx-swift-lm ne fait pas.
4. Le client Core Audio amont (Apache-2.0) est copié tel quel puis branché sur le moteur.

**Conséquences** : embarquable (Fluxforge Studio, iPhone à étudier) ; chaque optimisation du
catalogue s'applique derrière une porte chiffrée (`plan/13-optimisations.md`).
