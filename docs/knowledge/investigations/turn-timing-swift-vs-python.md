# Délai de réponse par tour : Swift contre haru_mlx (2026-11-18)

**Question.** Vincent, première conversation dans Haru Studio : « on a du mal à observer à quelle
injonction Haru répond ». Défaut du portage ou comportement du modèle ?

**Mesure.** Entrée fixe : 5 questions (`say`, anglais) espacées de 8 s, 54,6 s. Session hors ligne bf16,
carte `en` d'exemple, voix de test, graines 0-9, Swift (`haru run --stereo --transcript`) et Python
(`Scripts/reference/dump_turns.py`). Tours extraits par `haru turns` (seuil −40 dBFS par trame,
fin de segment après 5 trames de silence) : délai = début de la réponse − fin de la question.

| | réponses / 5 (moy.) | délai médian | délais > 3 s | chevauchements (< −0,5 s) | mots / session (médiane) |
|---|---|---|---|---|---|
| Swift | 3,9 | +0,53 s | 8 | 5 | 39,5 |
| haru_mlx | 3,8 | +0,53 s | 2 | 8 | 42 |

**Conclusion.** Même distribution : l'irrégularité (questions sans réponse, réponse 5-6 s plus tard
ou à la question précédente, monologue par-dessus l'utilisateur) vient du modèle échantillonné, pas
du portage. Graine 1 : même première phrase des deux côtés, puis divergence bf16 attendue. Le seul
écart (8 contre 2 délais > 3 s) est à revoir sur plus de graines si une plainte précise le vise.

**Suite.** L'app montre désormais deux pistes alignées et une chronologie par tour (délai affiché) ;
le journal de l'app consigne les tours et la décomposition de la latence audio. `haru simulate`
vérifie le chemin vivant des niveaux (409 trames, écart ≤ 2,2 dB au fichier, dû au passage 48 kHz).
