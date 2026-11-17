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
- D8 (Q1, 2026-11-17). Précision par défaut : **bf16** (16 bits, comme publié). Le 8 bits reste un
  profil mesuré (T-5.1/T-5.2) ; si le p99 bf16 dépasse la porte, la question revient à Vincent.
- D9 (Q2, 2026-11-17). Dépôt GitHub **public** dès maintenant (VincentGourbin/haru-swift-mlx).
  Conséquence : jamais de chemin local, de poids, d'extrait de voix ni d'audio généré dans le dépôt.
- D10 (Q3, 2026-11-17). Voix de test : un extrait local de discours public (203 s, MP3 44,1 kHz
  stéréo ; les 180 dernières secondes sont gardées), **hors dépôt**, désigné par `HARU_TEST_VOICE`.
  Usage de test local uniquement : aucun audio produit avec cette voix n'est publié.
- D11 (Q4, 2026-11-17). Jalon 6 : **app autonome** `haru-studio` ; l'intégration Fluxforge (T-6.4)
  reste une fiche ultérieure.

## Ouvertes
Aucune.
