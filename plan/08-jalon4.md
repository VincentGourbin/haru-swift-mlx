# 8. Jalon 4 — Conversation en temps réel (Core Audio dans le processus)

| Fiche | Contenu | Porte |
|---|---|---|
| T-4.1 | `MLXWorker` : un fil dédié (QoS userInteractive), file de travaux, échéance attendue + sonde de chauffe (`warm`, 30 ms par défaut) ; `HaruEngine` (actor public) qui ne laisse sortir aucun `MLXArray` | test : tout appel MLX vient du fil (assertion), annulation propre |
| T-4.2 | `HaruAudio` : copie de `Shared/Audio.swift` amont (périphériques, HAL units, `Resampler`, `Player`, `wsolaCompress`, `DuplexAudio`, rapport de latence), en-têtes Apache-2.0 | `haru devices` liste les périphériques ; boucle micro → haut-parleur sans modèle |
| T-4.3 | Pont temps réel : 48 → 24 kHz vers `feed` sur le fil MLX (tampon circulaire sans verrou), `step` dès `framesReady`, 24 → 48 kHz vers `Player` ; texte vers un `AsyncStream` | 10 min de « conversation » avec un fichier joué en entrée virtuelle : 0 sous-alimentation |
| T-4.4 | Mesure en ligne : pas par trame (horloge hôte), latence CAN→CNA, sous-alimentations, rattrapage ; rapport en fin de session + `WAV` stéréo enregistré | rapport cohérent avec celui du client amont sur la même session |
| T-4.5 | `haru talk --voice … --card … [--input --output --margin-ms --precision]` ; préfixe préparé en arrière-plan (clone d'état, T25) ⛔ G-3 | portes de `plan/03-budget.md` ; Vincent converse 10 min (filaire) |
