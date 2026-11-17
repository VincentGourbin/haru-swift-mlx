# Questions à Vincent

## ASK — T-0.1 — 2026-11-17
- Contexte : squelette validé ; poids téléchargés (10,54 Go) ; la mesure de référence (T-0.2) exige
  d'installer `haru_mlx` amont (pip, en editable) dans `.venv-ref` — refusé par le mode auto.
- Ce que j'ai essayé : `python3 -m venv .venv-ref && pip install -e reference/haru/haru_mlx` → refusé.
- Question : autorises-tu cette installation (ou la lances-tu : `! python3 -m venv .venv-ref && .venv-ref/bin/pip install -e reference/haru/haru_mlx`) ?
- Options : A) j'installe et je mesure ; B) tu l'installes ; C) on saute la mesure (déconseillé).
- Questions de G-0 en attente : Q1-Q4 de `plan/16-decisions.md`.

**Réponse de Vincent (2026-11-17)** : installation faite par Vincent ; bf16 par défaut ; dépôt public ;
voix de test = extrait local (`HARU_TEST_VOICE`, hors dépôt) ; app autonome. → `plan/16-decisions.md` D8-D11.
