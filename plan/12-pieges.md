# 12. Checklist de pièges (contrat de revue — cocher à chaque fiche)

Numéros cités par les fiches. « (amont) » = documenté dans haru_mlx ; « (cat. n) » = catalogue
`mlx-swift-audit` ; « (S) » = Sperid.

## Numérique
1. Gains RMSNorm du backbone : checkpoint `w` centré, modèle `1 + w` en **fp32** (amont).
2. GatedNorm et projecteurs audio : gain **non** recentré, pas de `1 + w`.
3. Noyau GDN : copier celui de haru_mlx ; celui de mlx-swift-lm a une somme de Kahan et des
   pragmas fp → résultats différents de la référence.
4. Mémoire GDN et `decay` en fp32, sur toute la conversation (la décroissance se compose).
5. Normalisation q/k du GDN : `rms_norm` sans gain avec `eps · scale²`, puis `scale²` (q) / `scale` (k).
6. RoPE : 64 dims, **non entrelacée**, appliquée après `q_norm`/`k_norm` ; MRoPE = 1-D pour du texte.
7. `qkv` de l'attention : le query de chaque tête est suivi de sa porte (`[B, T, H, 512]` puis split).
8. Logits PAD/EOP recalculés en fp32 : la décision parler/se taire est sous la résolution bf16 (amont).
9. Tête : ordre des sommes du conditionnement fixé ; `ScalarEmbed` arrondit la valeur au dtype d'abord.
10. Codec : conv strided exige kernel = 2·stride ; clip incomplet complété à droite ; filtres
    depthwise `[C, 7]` (taps dans l'ordre temporel).
11. Latents utilisateur et de référence **échantillonnés** (`codec.sample`), pas la moyenne.
12. `latent_scale` / `latent_bias` : scalaires fp32 du checkpoint, pas des constantes.

## MLX Swift
13. Fonctions compilées : elles capturent les poids → recompiler après quantification ou
    rechargement ; vider les échantillonneurs de la tête (amont).
14. Un seul fil MLX ; poids paresseux matérialisés sur ce fil (S : plantage entre threads).
15. Chargement : `eval` par partie, jamais `eval(model.parameters())` global ; pas de tenseur
    fusionné qui double les poids (cat. 24, R8).
16. `Memory.cacheLimit` fixé (amont : 1 Gio) sinon chaque agrandissement du KV laisse ses anciens
    tampons en pool ; `clearCache()` après le prompt (amont). Limite câblée = working set recommandé.
17. Pas de `.item()` / statistique dans le chemin chaud (cat. T28) ; un seul `eval` par trame (T24).
18. Deadlock ABBA compile × vjp : tests en série (cat. 20) — pas de gradient ici, mais la règle tient.
19. `swift build` ne trouve pas `default.metallib` : `xcodebuild` seulement.

## Temps réel
20. Mesurer à cadence vivante : à 7,5 Hz le GPU est inactif ~70 % du temps, son horloge baisse ;
    un banc dos à dos sous-estime le pas réel.
21. Juger au p99 et aux sous-alimentations, jamais à la moyenne ; la KV grandit : mesurer en fin de
    session longue.
22. Fils audio : aucune allocation, aucun MLX, aucun verrou bloquant.
23. Bluetooth : ≥ 100 ms de plus ; micro/HP intégrés : traitement de quelques dizaines de ms (amont).
24. Rééchantillonnage : `AVAudioConverter` en un appel a rendu du silence au suréchantillonnage
    (cat. 46) ; vérifier l'énergie de sortie dans un test.
25. Autre app MLX ou GPU active pendant un banc : `pgrep` avant chaque mesure (cat. 8) ;
    Spotlight : sorties dans `.noindex` (cat. 19).
26. Une voix clonée n'est utilisée qu'avec les droits : aucun extrait de voix dans le dépôt.
