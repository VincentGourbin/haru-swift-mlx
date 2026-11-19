# Journal

## 2026-11-17 — création du projet
- Lecture de `higurashi-ai/Haru` @ `7fe6b76` : haru_mlx (modèle, session, backbone Qwen3.5, codec VibeVoice, tête MeanFlow, noyaux Metal GDN et mélangeur depthwise, plan de poids), client Core Audio `swift/Shared/Audio.swift`, cadencement `worker.py` / `backends.py` / `native.py`.
- Poids téléchargés (`hf download`, 10,54 Go) sous `$HARU_MODELS_DIR/higurashi-ai/Haru-4B`.
- `haru info` (MESURE de l'en-tête) : 1022 tenseurs, 5 270 M paramètres ; embedding 1,27 Go, backbone 7,14 Go, tête 0,73 Go, codec 2 × 0,69 Go, projecteurs 0,03 Go. Soit ≈ 10,5 Go lus par trame en bf16 (CALCUL, `plan/03-budget.md`).
- Machine : M3 Max 14 CPU / 30 GPU, 96 Go — bande passante annoncée 300 Go/s contre 400 pour le M1 Max des chiffres amont (43 / 31 ms par trame) : le bf16 risque d'être plus serré ici (CALCUL ≈ 57 ms).
- Le GDN de mlx-swift-lm (vendored dans qwen38) diffère du noyau de haru_mlx (somme de Kahan) : piège n° 3.
- Mesure de référence Python reportée (installation de haru_mlx refusée par le mode auto, en attente de l'accord de Vincent : `tasks/ASK.md`).
- SwiftPM a résolu mlx-swift **0.32.3** (plage 0.31.6..<0.33.0), swift-transformers 1.4.0, swift-mlx-profiler 1.5.1.
- T-0.3 : plan de conversion porté, vérifié sur l'en-tête réel (898 paramètres, 0 écart) : les faits de `plan/02.6` sont confirmés par le checkpoint, pas seulement par lecture.
- Le générateur tiny amont (`make_tiny_checkpoint.py`) exige torch et le paquet PyTorch `haru` : T-1.1 écrira son propre générateur numpy.

## 2026-11-17 — T-0.2, mesure de référence
- haru_mlx sur M3 Max : bf16 vivant p50 55,7 ms (dos à dos 44,5), 8 bits 41,3 (30,6) ; `feed` + chauffe 44,7 / 29,3 ms ; pic MLX 11,2 / 6,8 Go. Le piège 20 est confirmé (+25 / +35 % en vivant).
- G-0 levée : bf16 par défaut (p99 vivant ≈ 63 ms, porte 90 ms), dépôt public, voix de test locale hors dépôt, app autonome.

## 2026-11-17 — T-0.4 et T-0.5
- T-0.4 : cartes des 9 langues et ids du system prompt identiques à haru_mlx (9 cartes + 8 chaînes limites).
- T-0.5 : aléa MLX partagé — clés, bf16, `categorical` identiques ; `normal` fp32 : 1 valeur sur 1152 à 1 ulp. mlx-swift 0.32.3 embarque le cœur MLX v0.32.2 (Python : 0.32.4), cause probable. Pas de rejeu (`decisions/rng.md`).
- T-0.6 : fichiers épinglés à la révision `8d6e83bc` du dépôt HF ; SHA-256 LFS du Hub = copies locales. **Jalon 0 terminé.**

## 2026-11-17 — Jalons 1 à 4 et application de test (session autonome, « rends-moi la main quand l'app de test est faite »)
- Parité sur les **vrais poids** contre des dumps haru_mlx (`Scripts/reference/dump_parity.py`, sous `.local-runs/parity`, jamais committés) au lieu des fixtures tiny : plus rapide, même machine. Décision D12.
- Codec : bf16 **bit-exact** ; fp32 encodeur 4,7e-7, décodeur 1,4e-5.
- Backbone + tête : fp32 ~1e-5 (niveaux amont), même top-1 ; bf16 non bit-exact — noyaux Metal compilés différemment entre MLX Swift et Python (`rms_norm` fp32 à 3,5e-8 sur entrées identiques ; pas la version : même écart avec mlx 0.32.2). Piège consigné.
- Session fp32 à graine 0 : 34 premières trames identiques à Python (même prompt de 1618 positions, même prise de parole), puis divergence d'échantillonnage attendue.
- Temps réel : voir `benchmarks/swift-live-m3max.md` — bf16 vivant 46,5 / 47,7 ms (Python 44,7 / 46,3), latence de réponse Core Audio 66 ms médiane.
- App `Haru Studio` : prête (prompt préparé en 2,5 s avec 180 s de voix). Piège : une app lit `~/Downloads` seulement après consentement TCC ; l'invite ne s'affiche pas écran éteint ; choisir la voix par « Choose… » l'accorde.
- Session de 5 min : le pas sautait de 46 à 57 ms après ~20 s (20 sous-alimentations) ; cause : l'horloge CPU baisse quand la pompe fait des siestes de 1 ms ; sonder en continu → 44,3 / 44,9 ms, latence 65 ms, 1 sous-alimentation. Python `--feed --warm-ms 30` y échappait parce qu'il attend activement en continu.
- Courses corrigées avant la première utilisation : fin de conversation (attendre la sortie de la pompe avant de lire ses mesures), démarrage pendant une préparation (attendre la préparation en cours).

## 2026-11-18 — Première conversation de Vincent ; chronologie des tours
- Journal de l'app : pas vivant 44,1 ms (p99 45), latence audio médiane 117 ms sur micro et haut-parleurs réels (65 ms sur périphérique virtuel), 1-2 sous-alimentations par session de 8-27 s, sans voix chargée. Le pas mesuré au chargement (47-51 ms) vaut la médiane vivante + 3 ms de marge, voulue comme en amont (vérifié plus tard le même jour).
- Plainte : on ne voit pas à quoi Haru répond. Mesure Swift contre haru_mlx sur 10 graines (`investigations/turn-timing-swift-vs-python.md`) : même distribution, comportement du modèle.
- Ajouts : `TurnDetector` (HaruCore/Analysis), `haru turns`, niveaux utilisateur par trame dans la pompe (`onUserLevel`), pistes et chronologie dans l'app, tours et décomposition de latence dans `studio.log`.

## 2026-11-18 — « Enchaîne tout ça » : réactivité, qualité, latence
- Rejeu forcé d'une session de 410 trames contre haru_mlx : fp32 à 1e-6 sans dérive, bf16 audio bit-exact et état caché plat à ~1e-2. L'écart de WER du juge ASR (16 % contre 6,5 %) est de la variance d'échantillonnage. Détails : `investigations/turn-timing-swift-vs-python.md`.
- Biais EOP : 0,5 dans l'app (4,5 réponses sur 5 au lieu de 3,9). Curseurs Eagerness et Temperature ; changer l'échantillonnage ne relance plus le prompt (`DuplexSession.retune`).
- Pas mesuré au chargement : 12 trames après 8 de chauffe (au lieu de 6 après 4) ; il vaut la médiane vivante + 3 ms par conception (amont), l'écart 47,7 / 44,8 ms est cette marge, pas une erreur.
- Rejeu fp32 : le jeton tiré par Swift (DRY, top-k, PAD/EOP fp32) est celui de Python aux 410 trames ; bf16 : 34 trames sur 410 diffèrent (bruit bf16).
- Latence audio : aucun gain ; les 117 ms du micro et des haut-parleurs réels restent à décomposer (journal de l'app, prochaine session).
- 8 bits : distance à fp32 double de celle du bf16 (état caché 4e-2 contre 2,1e-2) ; reste une option.
- `haru run --seeds N --eop-bias B` (une charge pour N graines ; isolation vérifiée, mêmes jetons qu'en processus séparés).

## 2026-11-18 (soir) — Sessions réelles de Vincent, suivies en direct
- Haut-parleurs du MacBook sans casque : Haru s'entend lui-même (transcription de ce que le modèle a reçu : sa propre voix + les « Stop » de Vincent) ; les segments « You » recopient ceux de Haru. Il faut un casque, ou l'annulation d'écho (voice processing) à ajouter.
- AirPods Max : passent à 24 kHz dès que le micro s'ouvre (liste à 48 kHz avant) ; le rééchantillonnage d'entrée suit, 47 983 éch./s arrivent au modèle. Latence audio 246 ms dont 178 ms de sortie Bluetooth ; micro de l'iPhone (Continuité) : +127 ms d'entrée.
- Avec casque, ce que le modèle a reçu est intelligible (Voxtral : « Attends… Raconte-moi une blague. Stop. ») mais Haru ne suit pas la demande. haru_mlx sur **le même enregistrement** (carte fr, sans voix, 4 graines) : même comportement (présentation, monologue, aucune blague). Comportement du modèle de base, pas du portage.
- App : périphériques relus à chaque démarrage de conversation ; journal en direct (tours, niveaux) ; `last-user-heard.wav` (ce que le modèle a reçu, dossier de logs local) ; fréquences réelles des périphériques.

## 2026-11-18 (soir) — Son fiabilisé, 8 bits mesuré
- Entrée brute : le format client suit la fréquence du périphérique une fois l'unité démarrée (reconfiguration si l'ouverture du micro la change : Bluetooth) ; tampons et latences relus après démarrage (les 246 ms AirPods étaient calculées avec 48 kHz au lieu de 24 kHz : à remesurer).
- Annulation d'écho : une unité VoiceProcessingIO (micro bus 1, lecture bus 0), AGC coupé, atténuation des autres sons minimale ; option dans l'app et `haru talk --echo-cancellation`. `haru echo-test` (joue un extrait, mesure ce qui revient) : sur le périphérique virtuel Teams (bouclage), brut +75 dB au-dessus du plancher, avec annulation médiane −120 dBFS. Haut-parleurs réels et parole simultanée : à mesurer avec Vincent.
- Périphériques : suivi des branchements et des défauts système (DeviceWatcher) ; seuil de détection de l'utilisateur adaptatif (plancher + 15 dB, entre −62 et −40) ; enregistrement de ce que le modèle entend en option ; bilan de fin : écho (niveau du micro pendant que Haru parle), mémoire.
- 8 bits (simulate, sans voix) : pas 30,5 / 31,8 ms (bf16 44,2 / 46,9), MLX actif 6,6 Go (bf16 10,5), pic 10,8 Go au chargement (bf16 chargé puis quantifié : un pack préquantifié l'éviterait).

## 2026-11-19 — écho mesuré en vrai, décompte des tours, dérivés sur HF
- Session réelle haut-parleurs + micro du MacBook : sans annulation d'écho, le micro est à +35 dB quand Haru parle (il s'entend, 0/3 tours) ; avec, −1 dB (écho effacé), 16 réponses, délai médian +0,53 s, la voix de l'utilisateur passe (pics −22 dBFS). Latence audio 114 ms stable.
- Décompte des tours : une phrase de Haru ne répond plus qu'à une intervention, commencée avant elle (parler par-dessus une phrase déjà lancée est un chevauchement, plus une « réponse » à délai négatif).
- Relevé des dérivés de Haru sur Hugging Face : aucun finetune ni LoRA, des conversions (dont le constat de cortiq : 4 bits casse le modèle), une évaluation communautaire, des datasets de tour de parole, et trois articles de « modèle qui souffle » (MoshiRAG, Context Spanning, SALMONN-duo), tous réentraînés. Voir `investigations/derives-haru-hf.md`.
