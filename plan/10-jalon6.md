# 10. Jalon 6 — App, API, intégrations, iPhone (étude)

| Fiche | Contenu | Porte |
|---|---|---|
| T-6.1 | API publique `HaruEngine` : `load(profile:)`, `prepare(card:voice:)`, `conversation() -> AsyncStream<Event>` (texte, niveau, latence), arrêt ; documentation `docs/API.md` | exemple compilé dans les tests |
| T-6.2 | `haru-studio` (SwiftUI macOS) : carte (9 langues, exemples amont), voix (fichier ou enregistrement 3 s-3 min), réglages (périphériques, guidance, marge, précision), deux pistes d'onde sur une horloge, transcription au rythme de l'audio, latence affichée | Vincent converse dans l'app ⛔ G-4 |
| T-6.3 | Export de la conversation : WAV stéréo, vidéo MP4 des pistes (comme l'amont) | fichiers lisibles |
| T-6.4 | Intégration Fluxforge Studio : résolution de dépendances à côté de flux-2 / yue2 / gemma (mlx-swift 0.31.6 exact) | build de Fluxforge vert |
| T-6.5 | Étude iPhone (15 Pro Max, 8 Go) : 8 bits ≈ 6,6 Go de poids lus par trame — tenue en mémoire, bande passante (≈ 50 Go/s → ≥ 130 ms : **probablement hors temps réel**), pistes (4 bits mixte, tête de sortie restreinte) | rapport `investigations/iphone.md`, décision |
