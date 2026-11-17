# 0. Mode d'exécution

- Le plan est exécuté fiche par fiche (`tasks/T-x.y.md`) par un agent qui suit `AGENTS.md`. Une fiche
  lit **un** fichier de `plan/`, crée au plus quelques fichiers Swift de 200 lignes, et se termine
  par des validations chiffrées (`BUILD OK`, `TESTS OK`, `PARITY OK`, et à partir du Jalon 4 des
  portes temps réel en p99).
- Les fiches d'un jalon n'ont leur fichier `T-x.y.md` qu'à l'ouverture du jalon ; `tasks/INDEX.md`
  porte l'ordre et l'état de toutes les fiches.
- ⛔ GATE : Vincent vérifie lui-même (écoute, conversation), répond dans `tasks/ASK.md`, puis le
  jalon suivant s'ouvre.
- Référence numérique : `reference/haru/haru_mlx` (Python MLX, même machine). Référence de
  comportement temps réel : `reference/haru/haru_mlx/swift/` (client Core Audio amont) et
  `native.py` / `backends.py` / `worker.py` (cadencement côté modèle).
- Copier d'abord, adapter ensuite : la structure et les noms de `haru_mlx` sont repris tels quels
  (paramètres fusionnés compris), pour que la parité se fasse tenseur par tenseur.
