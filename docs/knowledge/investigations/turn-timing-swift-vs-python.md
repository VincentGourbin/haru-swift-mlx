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

**Conclusion (timing).** Même distribution : l'irrégularité (questions sans réponse, réponse 5-6 s plus tard
ou à la question précédente, monologue par-dessus l'utilisateur) vient du modèle échantillonné, pas
du portage. Graine 1 : même première phrase des deux côtés, puis divergence bf16 attendue. Le seul
écart (8 contre 2 délais > 3 s) est à revoir sur plus de graines si une plainte précise le vise.

**Suite.** L'app montre désormais deux pistes alignées et une chronologie par tour (délai affiché) ;
le journal de l'app consigne les tours et la décomposition de la latence audio. `haru simulate`
vérifie le chemin vivant des niveaux (409 trames, écart ≤ 2,2 dB au fichier, dû au passage 48 kHz).

## Qualité audio : juge ASR, puis rejeu forcé (2026-11-18)

Juge : Voxtral Mini Realtime 4B 6 bits (`convertvoxtral`, `--delay 960`) transcrit la voix de Haru,
comparée à son propre flux de texte (WER, nombres et hésitations normalisés), 10 graines par condition :
Swift bf16 16,0 %, haru_mlx bf16 6,5 %, Swift fp32 10,4 %, Swift 8 bits 18,9 %, biais EOP 0,5 12,2 %.
Écart dominé par une ou deux conversations (Swift graine 6 : 38 erreurs sur 69 ; fp32 graine 5 : 24 sur 43) :
le juge mesure surtout ce que le modèle a choisi de dire, il n'est pas assez puissant à 10 graines.

Test décisif, déterministe (`RealSessionReplayTests`, dumps `Scripts/reference/dump_replay.py`) : une
session haru_mlx de 410 trames (graine 5, biais EOP 1) rejouée en Swift en imposant à chaque trame le
jeton et le latent de Python.

| | état caché | latent de la tête | latent utilisateur | audio |
|---|---|---|---|---|
| fp32 contre fp32 (médiane, toutes trames) | 1e-6 | 2e-6 | 1,5e-6 | 6e-7 à 5e-6 |
| bf16 contre bf16 | 1,1e-2 à 1,7e-2, plat | 3e-2 | 0 | 0 (bit-exact) |
| bf16 Swift contre fp32 Python | 2,1e-2 | 3,9e-2 | 2,3e-2 | 1,6e-2 |
| 8 bits Swift contre fp32 Python | 4,0e-2 | 8,1e-2 | 2,3e-2 | 1,6e-2 |

Aucune dérive sur 410 trames : codec en flux, cache KV, réinjection du latent sont exacts. L'écart
de WER n'est donc pas un défaut du portage. Le 8 bits double la distance à fp32 du bf16 : admis
comme option (plus rapide, −4 Go), pas comme défaut sans écoute.

## Biais EOP (« Eagerness » dans l'app), Swift bf16, 10 graines

| biais | réponses / 5 | délai médian | chevauchements | mots / session |
|---|---|---|---|---|
| 0 | 3,9 | +0,53 s | 5 | 40 |
| 0,5 | 4,5 | +0,40 s | 11 | 46 |
| 1 | 4,7 | −0,67 s | 24 | 77 |
| 2 | 4,8 | −3,47 s | 35 | 169 |
| 3 | 5,0 | 0,00 s | 24 | 167 |

0,5 devient la valeur par défaut de l'app (la bibliothèque garde 0, comme l'amont) ; au-delà de 1,
Haru monologue.
