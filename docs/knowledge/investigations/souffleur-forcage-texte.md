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
