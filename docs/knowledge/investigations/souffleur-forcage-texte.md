---
type: investigation
date: 2026-11-19
status: première mesure
---

# Souffleur : Haru prononce-t-il les mots qu'on impose à son flux texte ?

Question décisive pour un cadre où un LLM choisit quoi dire et Haru le dit (voir `derives-haru-hf.md`).

## Méthode

`haru run … --steer "<phrase>" --steer-after <s> [--steer-when eop|now] [--steer-pace fixed|model]` :
`DuplexSession.textOverride` remplace le jeton texte échantillonné, le latent audio reste celui du modèle.

- `eop` : on attend que Haru prenne lui-même la parole (jeton EOP), puis on impose la phrase.
- `now` : on force l'EOP à l'instant voulu, puis la phrase.
- Un jeton imposé par frame (le rythme naturel de Haru), puis 3 s de silence texte, puis Haru reprend librement.
- Juge : Voxtral Realtime 6 bits sur la piste de Haru, fenêtre de quelques secondes après l'injection ;
  score = rappel des mots de la phrase, dans l'ordre (plus longue sous-suite commune).
- bf16, sans voix de référence, EOP +0,5, seeds 0 à 4, une injection après la 3ᵉ intervention.

## Résultats

**Anglais** (enregistrement de test anglais, carte `en`), « Did you know that octopuses have three hearts and
blue blood? » :

| condition | rappel par seed | moyenne |
|---|---|---|
| sans injection (même fenêtre) | 9 · 9 · 27 · 18 · 9 % | 14 % (mots courants) |
| `eop` | 55 · 100 · 100 · 64 · 100 % | 84 % |
| `now` | 91 · 91 · 100 · 100 · 100 % | 96 % |

La voix commence 0,13 à 0,27 s après le premier jeton imposé (1 à 2 frames). Après la phrase, Haru enchaîne
de lui-même sur le sujet (« Do you know why… », « Yeah, I think that's so cool »).

**Français** (enregistrement synthétique français fait avec `say -v Jacques`, carte `fr`) :

| phrase | rappel par seed |
|---|---|
| « Tu savais que les pieuvres ont trois cœurs et le sang bleu ? » | 0 · 42 · 8 · 42 · 83 % |
| « Tu veux que je te raconte une blague ? » | 0 · 100 · 38 · 0 · 100 % |

- Seed 0 : le texte est imposé mais **la voix reste muette** (aucune énergie pendant 7 s). Le latent audio
  peut ne pas suivre le texte.
- Le reste du temps la voix parle, mais Voxtral l'entend mal : la parole libre de Haru en français est déjà
  peu intelligible pour lui (« j'ai 30 de croissant » pour « j'ai vingt-deux ans », des passages pris pour de
  l'allemand). Le juge et la voix par défaut sont en cause autant que le forçage.
- Avec l'enregistrement anglais et la carte française, c'était pire (0 à 25 %) : la langue de l'interlocuteur
  compte.
- Laisser Haru rythmer (`--steer-pace model`, ses PAD et EOP conservés) n'améliore rien.

**Français avec une voix de référence française** (synthétique, `say -v Flo`, 25 s) : la parole libre de Haru
devient intelligible (« je m'appelle Haru, j'ai 22 ans et je finis ma licence »), et le forçage suit :

| phrase | rappel par seed |
|---|---|
| pieuvres | 0 (voix muette) · 92 · 75 · 92 · 0 (voix présente, Voxtral vide) % |
| blague | 12 · 88 · 100 · 100 · 62 % |

Le seed 0 reste muet dans les deux réglages français : le cas « texte imposé, voix muette » est reproductible.

## Conclusion

En anglais, **le forçage du flux texte fait dire à Haru la phrase voulue, au moment voulu** (`now`) et avec sa
voix : le cadre « un LLM décide, Haru parle » est faisable sans entraînement. En français, il faut
une voix de référence française : alors 6 essais sur 10 sont dits à 75 % ou plus, avec des confusions de mots
proches (« sœurs » pour « cœurs »). Dans 1 à 2 essais sur 10 la voix ne suit pas le texte.

À faire ensuite : détecter le cas « texte imposé, voix muette » (énergie de sortie basse après l'EOP forcé) et
réessayer ; un juge plus solide en français (Voxtral fp16) ; puis brancher un vrai LLM qui écoute la
transcription et décide quoi dire et quand.

## Suite (même jour) : relance quand la voix reste muette, puis boucle complète avec un décideur

**Voix muette.** `Prompter` (HaruCore) remplace l'injection ponctuelle : file de répliques, EOP puis jetons,
détection de mutisme (8 frames sous −50 dBFS après l'EOP), relance, abandon si l'utilisateur parle.
Sur le cas reproductible (français, seed 0), la relance ne suffit pas : relancer tout de suite, après 0,67 s
de pause, avec un « Ah, » d'amorce ou avec un guidage audio à 1,0 pendant la relance, la voix reste muette
à 24,8 s et à 28 s, mais parle si on injecte à 25,6 s ou à 26,5 s. Le mutisme dépend de l'état de la
session, pas du hasard : il est détecté et signalé, une seule relance est gardée.

**Boucle complète hors ligne** (`Scripts/prompter/offline.py`) : fin de l'intervention par le même seuil que
`TurnDetector` → transcription Voxtral Realtime de l'intervention seule (1 s de contexte avant, 1 s de
silence après) → Gemma 4 E4B 4 bits en local (`gemma4-server`, rien ne sort du Mac) → réplique due à
fin + temps ASR + temps LLM mesurés (causal). Haru la dit via `--steer-plan`. Temps mesurés : ASR 0,7 à
1,6 s (outil en ligne de commande, l'intervention entière), LLM 0,1 à 0,45 s. La voix démarre donc 1,2 à
2,3 s après la fin de l'utilisateur, contre 0,5 à 1 s pour Haru seul ; une transcription en flux
ramènerait la partie ASR vers 0,3 s.

« Dite » = rappel des mots de la réplique ≥ 50 % sur la piste de Haru (Voxtral) ; ça mesure si la voix a dit
la phrase imposée, pas si la phrase répondait bien.

| | répliques dites | rappel moyen par seed |
|---|---|---|
| anglais, ASR réel, **d'un bloc** | **24/25** | 100 · 75 · 100 · 96 · 100 % |
| anglais, ASR réel, découpé par phrases | 18/25 | 23 · 62 · 99 · 46 · 96 % |
| français, transcription parfaite, d'un bloc | 9/25 | 60 · 6 · 55 · 10 · 35 % |
| français, transcription parfaite, découpé par phrases | 8/25 | 55 · 13 · 33 · 14 · 48 % |

- **Le découpage par phrases casse la voix** : avec le texte frame par frame (seed 0), la voix dit le premier
  morceau puis se tait après la pause insérée, alors que le texte continue (« … this morning. » dit,
  « It was pretty quick. » muet). D'un bloc, 92 % des jetons de mots imposés sont voisés sur 5 seeds. Le
  découpage est désactivé par défaut (`--steer-chunked` pour l'essayer).
- Le décideur répond à la demande quand il l'a bien entendue : « Raconte-moi une blague » → une blague,
  « une idée de recette » → des pâtes aux légumes. En anglais, la transcription n'a compris que 3 des
  5 interventions ; pour les deux autres, Gemma répond à ce qu'il a cru entendre (« Sorry, I didn't quite
  catch that », « Okay… goodbye then ») : dites, mais pas de vraies réponses.
- Haru en train de parler au moment d'imposer n'est pas la cause des échecs : 55 % de répliques dites
  quand Haru parlait dans la seconde d'avant, 37 % quand il se taisait (75 répliques, toutes conditions).
  Ça rejoint le cas muet : c'est plutôt l'état « j'écoute » qui résiste.
- En français, une partie des répliques sort inintelligible même d'un bloc et avec la voix de référence
  française.
- La transcription de l'utilisateur est le maillon faible : Voxtral Realtime n'a pas d'option de langue et
  bascule (« Telme Abou Atri Piyoudé Loftoté », « I call it black » pour « Raconte-moi une blague »). Les
  versions 6 bits (`mlx-community`) et 8 bits (`ellamind`), deux dossiers distincts, donnent le même texte :
  le juge est cohérent, ce qui ne prouve pas qu'il ait raison. La reconnaissance d'Apple (sur l'appareil, en
  flux, langue fixée, sans la carte graphique de Haru) demande une autorisation qu'un outil en ligne de
  commande n'obtient pas ; à essayer dans l'app.

**Bilan.** En anglais, imposer une réplique entière d'un bloc fait dire à Haru 24 répliques sur 25, de 9 à
15 mots, 1,9 s après la fin de l'utilisateur (dont 1,4 s de transcription par outil en ligne de commande et
0,35 s de LLM). Restent : une transcription en flux à langue fixée, le français, et la mesure du pas de
Haru quand l'ASR et le LLM tournent en même temps sur la carte graphique, avant de passer dans l'app.
