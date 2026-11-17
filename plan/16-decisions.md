# 16. Décisions actées et questions ouvertes

## Actées (2026-11-17)
- D1. Nom : `haru-swift-mlx`, paquet `HaruSwift`, CLI `haru`. Dépôt git **local**, sans remote.
- D2. Port 100 % Swift dans le processus ; pas de serveur Python, pas de socket.
- D3. Référence numérique = haru_mlx (Python MLX) sur la même machine, pas PyTorch ; structure
  et paramètres fusionnés repris à l'identique.
- D4. Pas de dépendance à mlx-swift-lm (copie et adaptation ; GDN de haru_mlx, piège 3).
- D5. Client Core Audio amont copié (Apache-2.0, en-têtes), pas réécrit.
- D6. Mesure de référence avant tout code de modèle (T-0.2, G-0).
- D7. Poids sous `$HARU_MODELS_DIR/higurashi-ai/Haru-4B` (téléchargés le 2026-11-17, 10,54 Go).

## Ouvertes
- Q1 (G-0). Précision par défaut : bf16 ou 8 bits ? (dépend de T-0.2)
- Q2 (G-0). Dépôt GitHub : privé, public, ou local seulement pour l'instant ?
- Q3. Voix de test : quel extrait (droits) pour les bancs et GATE ? Un enregistrement de Vincent ?
- Q4. Jalon 6 : app autonome `haru-studio` ou onglet dans Fluxforge Studio d'abord ?
