---
type: investigation
date: 2026-11-19
status: fait
---

# Dérivés de Haru-4B sur Hugging Face, et pistes pour un « souffleur »

Relevé du 2026-11-19 (une semaine après la sortie de Haru-4B, le 2026-11-12).

## Ce qui existe

**Aucun finetune ni LoRA de Haru-4B.** Les seuls dépôts étiquetés `finetune` sont des modèles aléatoires
minuscules pour les tests (`tiny-random/haru`, `yujiepan/haru-tiny-random`). higurashi-ai ne publie ni code
d'entraînement ni dataset (le dépôt GitHub ne contient que l'inférence : `haru`, `haru_mlx`, `haru_vllm`).
Leur autre modèle, `higurashi-ai/FastDialogueSidon`, sépare une conversation enregistrée en deux pistes
par locuteur : c'est très probablement l'outil qui a fabriqué les données d'entraînement de Haru.

**Conversions** (même modèle, autre moteur) :

| dépôt | quoi | à retenir |
|---|---|---|
| `sakasegawa/Haru-4B-GGUF` | GGUF llama.cpp | le plus téléchargé (26 k) |
| `infosave/Haru-4B-cmf` | moteur Rust `cortiq` | 8 bits pour le LM, f16 pour la tête et le codec ; **4 bits casse le modèle** (le flux texte devient du bruit). 84 ms/frame sur M4. |
| `kabcnd/Kenzan-Haru-4B-PinGrid-GGUF` | quantification mixte 3,8 Gio | protège les poids aberrants (embedding lié en Q6_K, `blk.3.attn_q` en Q8_0, tête MeanFlow en bf16) ; vise un Mac 16 Go. Seule la perplexité texte est mesurée, pas la conversation. |
| `nano16/Haru-4B-mlx-8bit` | MLX 8 bits | |

Pour le Jalon 5 : notre int8 (6,6 Go actifs) rejoint le constat de cortiq ; un 4 bits uniforme est à
proscrire, un mixte « protégé » reste possible mais à mesurer sur la conversation, pas en perplexité.

**Spaces** :

- `hugging-apps/haru-4b`, `akhaliq/haru-4b-studio` : démos hors ligne (un clip en entrée, la réponse en
  sortie), carte de personnage choisie par langue, pas de temps réel.
- `davidxifeng/haru-4b-duplex-eval` : **la seule évaluation publique.** Sessions de 3 à 5 min (cours de
  japonais, aide aux devoirs, coach d'anglais, écoute, jeu de rôle), tests de pause, d'acquiescement et
  d'interruption, et un A/B « long prompt libre » contre « carte structurée » (`card_templates.json`).
  Leurs transcriptions confirment ce qu'on observe : prise de parole et acquiescements corrects, contenu
  improvisé, calculs faux (« 12 + 19 = 31 ? » → « non, ça va »). La carte structurée tient mieux le rôle
  que le prompt libre (le barista japonais répond à la commande, le coach d'anglais pose des questions sur
  le métier), sans pour autant suivre des consignes.

## Datasets utilisables plus tard

- `Dayuxiaoshui/Ling-3.0-tiny-Full-Duplex-Brain-SFT` (MIT, 32 k points synthétiques) : un **« cerveau »
  de tour de parole** qui choisit `LISTEN / BACKCHANNEL / SPEAK / STOP / IGNORE` à partir du texte reconnu,
  de la durée du silence et de ce que l'agent est en train de dire. C'est exactement la moitié « quand »
  d'un souffleur ; la moitié « quoi » manque.
- `jatshi/trusted-full-duplex-agent-data` : reprise de contexte après interruption (« où en étais-tu ? »),
  avec un MLP de tour de parole sur des frames de 200 ms.
- `otoearth/otoSpeech-full-duplex-*` (280 h, dont 20 h « orientées tâche ») : vraies conversations
  stéréo ; accès sur demande. Ce serait la matière d'un finetune de Haru, si on en fait un.

## Travaux qui font déjà « un modèle qui souffle »

Trois articles décrivent l'architecture visée : un modèle duplex rapide qui parle, et un LLM
lent qui réfléchit à côté et lui transmet quoi dire.

- **MoshiRAG** (2604.12928, Kyutai) : Moshi émet un jeton « je dois chercher » ; un LLM répond en
  arrière-plan ; la réponse est compressée et ajoutée au flux d'entrée. Le modèle comble le délai avec une
  amorce (« alors, voyons… ») avant de dire le contenu.
- **Context Spanning** (2609.33443, Mindlogic) : même schéma, mais le texte du LLM est **injecté tel quel**
  par un préremplissage d'un bloc de positions, calculé dans le budget d'une frame. Ils notent que mêler
  l'information au flux de l'utilisateur fait croire au modèle que l'utilisateur l'a dite. Code publié.
- **SALMONN-duo** (2609.34247) : « système 1 » duplex, « système 2 » LLM agent ; le système 1 apprend
  quand déléguer et comment intégrer la réponse sans casser le fil.

**Point commun : les trois réentraînent le modèle duplex** pour qu'il sache lire l'information injectée.
Avec Haru de base, non entraîné à cela, il faut d'abord vérifier ce qui marche sans entraînement.

## Ce que ça donne pour notre cadre

Trois manières de souffler à Haru, de la plus directe à la plus douce :

1. **Forcer le flux texte.** Le LLM écrit la réponse ; on impose ces jetons à Haru frame par frame (le
   crochet `forced` du replay le fait déjà). Haru choisit le moment de parler (EOP), le souffleur choisit
   les mots. Question décisive : la voix prononce-t-elle les mots imposés ? Dans Haru, le jeton texte et le
   latent audio sortent du même état caché, et le jeton choisi entre dans la frame suivante : c'est
   plausible, mais à mesurer (transcription Voxtral de la voix contre le texte imposé).
2. **Injecter un contexte** (à la Context Spanning, sans entraînement) : préremplir quelques positions de
   texte (« [note : l'utilisateur demande une blague] ») et laisser Haru reformuler. Plus naturel si ça
   marche, mais le modèle de base risque de l'ignorer ou de le lire à voix haute.
3. **Changer la carte entre deux tours** : la carte structurée oriente le rôle (constat du Space
   d'évaluation) ; la remplacer coûte un nouveau préremplissage du préfixe.

Pour la décision « quand intervenir », le dataset Ling Brain donne un format tout fait (texte ASR +
silence + état de l'agent → action), et notre `TurnDetector` et Voxtral fournissent déjà ces entrées.

## Sources

- https://huggingface.co/infosave/Haru-4B-cmf
- https://huggingface.co/kabcnd/Kenzan-Haru-4B-PinGrid-GGUF
- https://huggingface.co/spaces/davidxifeng/haru-4b-duplex-eval
- https://huggingface.co/datasets/Dayuxiaoshui/Ling-3.0-tiny-Full-Duplex-Brain-SFT
- https://huggingface.co/datasets/jatshi/trusted-full-duplex-agent-data
- https://huggingface.co/higurashi-ai/FastDialogueSidon
- https://huggingface.co/papers/2604.12928 (MoshiRAG)
- https://huggingface.co/papers/2609.33443 (Context Spanning) — https://github.com/mindlogic-ai/ContextSpanning
- https://huggingface.co/papers/2609.34247 (SALMONN-duo)
