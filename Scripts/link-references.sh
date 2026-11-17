#!/usr/bin/env bash
# Crée reference/ : liens symboliques vers les dépôts frères (sources à copier) et clone épinglé
# de la référence amont. Tout est gitignoré ; commits épinglés dans plan/14-ecosysteme.md.
set -euo pipefail
cd "$(dirname "$0")/.."
DEV="${HARU_SIBLINGS_DIR:-$(cd .. && pwd)}"
mkdir -p reference
link() { [ -e "reference/$1" ] || ln -s "$DEV/$2" "reference/$1"; }
link sperid   sperid-swift-mlx
link yue2     YuE2-mlx-swift
link qwen38   qwen38-mlx-swift
link voxtral  convertvoxtral
link profiler swift-mlx-profiler
clone() {  # clone <dir> <url> <commit>
  if [ ! -d "reference/$1" ]; then
    git clone --quiet "$2" "reference/$1"
  fi
  git -C "reference/$1" checkout --quiet "$3"
  echo "$1 @ $(git -C "reference/$1" rev-parse --short HEAD)"
}
# higurashi-ai/Haru : haru_mlx (référence numérique), swift/ (client Core Audio), haru/ (PyTorch).
clone haru https://github.com/higurashi-ai/Haru.git 7fe6b76
ls -l reference
