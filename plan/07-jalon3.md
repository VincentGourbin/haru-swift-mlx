# 7. Jalon 3 — Tête MeanFlow, prompt, session, conversation hors ligne

| Fiche | Contenu | Porte |
|---|---|---|
| T-3.1 | `ScalarEmbed`, `HeadLayer`, `FinalLayer`, `MeanFlowHead` + échantillonneur compilé par (pas, guidance, dtype) | `PARITY OK head` (2e-6 fp32) avec bruit fixé |
| T-3.2 | `sampleToken` (top-k par `argPartition`, top-p sur les k), `DryState` (port exact de `dry_penalties`), `StreamDetokenizer` | tests portés de `tests/test_sampling.py` |
| T-3.3 | `PromptBuilder` : carte, référence (latents échantillonnés), `speak_first`, ton de pièce, latent de silence | positions/masques identiques à `build_prompt` sur 3 cas |
| T-3.4 | `DuplexSession` : `start` (prefill par 512, `clearCache`), `feed`, `framesReady`, `step`, `emit` (un seul `eval` par trame) | 50 trames à graine fixe : jetons identiques à haru_mlx (fp32), audio ≤ tolérance |
| T-3.5 | `haru run user.wav reply.wav --voice voice.wav --card card.json` + sortie stéréo (user G / modèle D) ⛔ G-2 | même transcription que `run_inference` (graine fixée) ; Vincent écoute |
