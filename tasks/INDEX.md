# Fiches — ordre imposé (une fiche = un commit = une validation)

État : `à faire` · `en cours` · `validée` · `bloquée`. Une fiche ne démarre que si tous ses prérequis sont `validée`. ⛔ = se termine par une question dans `tasks/ASK.md`. Le détail (porte) est dans le fichier de jalon cité ; une fiche n'a son fichier `T-x.y.md` qu'au moment où son jalon s'ouvre.

| Fiche | Titre | Jalon (plan) | Prérequis | État |
|---|---|---|---|---|
| T-0.1 | Squelette, scripts, configuration, inventaire, `haru info` | `plan/04-jalon0.md` | — | validée |
| T-0.2 | Environnement Python de référence + mesure de référence bf16 / 8 bits à cadence vivante ⛔ G-0 | `plan/04-jalon0.md` | T-0.1 | validée (G-0 levée 2026-11-17) |
| T-0.3 | `WeightPlan` (port de `weights.py`) + `haru info --plan` | `plan/04-jalon0.md` | T-0.1 | validée |
| T-0.4 | Tokenizer + rendu de carte + ids du system prompt | `plan/04-jalon0.md` | T-0.2 | validée |
| T-0.5 | `NoiseSource` + fixture d'aléa Python ↔ Swift | `plan/04-jalon0.md` | T-0.2 | validée |
| T-0.6 | `haru download` (copie de `HubDownloader`) + SHA-256 épinglés | `plan/04-jalon0.md` | T-0.3 | validée |
| T-1.1 | Fixtures tiny + `check-fixtures.sh` / `check-parity.sh` | `plan/05-jalon1.md` | T-0.2 | validée (2026-11-17, D12/D13) |
| T-1.2 | Convolutions causales (matrice) | `plan/05-jalon1.md` | T-1.1 | validée (2026-11-17, D12/D13) |
| T-1.3 | Mélangeur depthwise (noyau Metal + repli), `Block` | `plan/05-jalon1.md` | T-1.2 | validée (2026-11-17, D12/D13) |
| T-1.4 | Encodeur, décodeur, clip entier | `plan/05-jalon1.md` | T-1.3 | validée (2026-11-17, D12/D13) |
| T-1.5 | `CodecStream` (feed / pop / decode) | `plan/05-jalon1.md` | T-1.4 | validée (2026-11-17, D12/D13) |
| T-1.6 | Poids réels, coût par trame, aller-retour wav ⛔ G-1 | `plan/05-jalon1.md` | T-1.5, T-0.3 | validée (2026-11-17, D12/D13) |
| T-2.1 | RMSNorm, GatedNorm, MLP, AudioProjector | `plan/06-jalon2.md` | T-1.1 | validée (2026-11-17, D12/D13) |
| T-2.2 | Règle delta (noyau haru_mlx + repli) | `plan/06-jalon2.md` | T-2.1 | validée (2026-11-17, D12/D13) |
| T-2.3 | GatedDeltaNet, Attention, KVCache | `plan/06-jalon2.md` | T-2.2 | validée (2026-11-17, D12/D13) |
| T-2.4 | Couches, modèle, chemin une position compilé | `plan/06-jalon2.md` | T-2.3 | validée (2026-11-17, D12/D13) |
| T-2.5 | Chargeur + `HaruModel` (embed, logits) | `plan/06-jalon2.md` | T-2.4, T-0.3 | validée (2026-11-17, D12/D13) |
| T-2.6 | Poids réels : parité prompt + pas | `plan/06-jalon2.md` | T-2.5 | validée (2026-11-17, D12/D13) |
| T-3.1 | Tête MeanFlow | `plan/07-jalon3.md` | T-2.6 | validée (2026-11-17, D12/D13) |
| T-3.2 | Échantillonnage, DRY, détokeniseur | `plan/07-jalon3.md` | T-0.4 | validée (2026-11-17, D12/D13) |
| T-3.3 | `PromptBuilder` | `plan/07-jalon3.md` | T-1.6, T-0.4 | validée (2026-11-17, D12/D13) |
| T-3.4 | `DuplexSession` | `plan/07-jalon3.md` | T-3.1, T-3.2, T-3.3, T-0.5 | validée (2026-11-17, D12/D13) |
| T-3.5 | `haru run` wav → wav ⛔ G-2 | `plan/07-jalon3.md` | T-3.4 | validée (2026-11-17, D12/D13) |
| T-4.1 | `MLXWorker` + `HaruEngine` | `plan/08-jalon4.md` | T-3.5 | validée (2026-11-17, D12/D13) |
| T-4.2 | `HaruAudio` (copie du client Core Audio amont) | `plan/08-jalon4.md` | T-0.1 | validée (2026-11-17, D12/D13) |
| T-4.3 | Pont temps réel micro → session → lecture | `plan/08-jalon4.md` | T-4.1, T-4.2 | validée (2026-11-17, D12/D13) |
| T-4.4 | Mesures en ligne, rapport, WAV stéréo | `plan/08-jalon4.md` | T-4.3 | validée (2026-11-17, D12/D13) |
| T-4.5 | `haru talk` + préfixe préparé ⛔ G-3 | `plan/08-jalon4.md` | T-4.4 | validée (2026-11-17, D12/D13) |
| T-5.1 … T-5.6 | 8 bits + pack, profils, A/B compilé/noyaux, recouvrement, investigations quantif, endurance 60 min | `plan/09-jalon5.md` | T-4.5 | à faire |
| T-6.1 … T-6.5 | API, haru-studio ⛔ G-4, export, Fluxforge, étude iPhone | `plan/10-jalon6.md` | T-5.2 | T-6.1 à T-6.3 : première version 2026-11-17 (`HaruEngine`, Haru Studio, WAV) ; G-4 posée |
