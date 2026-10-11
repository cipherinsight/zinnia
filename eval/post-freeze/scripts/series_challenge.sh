#!/usr/bin/env bash
# One scaling series of the challenge constructions, points in sequence (as in
# eval-redo/layer/layer-challenge-data/run_scaling.sh), plus the compiler's own choice ("auto").
# Usage: series_challenge.sh WT OUT FAMILY [--time]     FAMILY = argsort | matmul | convolve
set -uo pipefail
WT=$1 OUT=$2 FAMILY=$3; shift 3
S=$(cd "$(dirname "$0")" && pwd)
PY=$WT/.venv/bin/python
run() { "$PY" "$S/scale_challenge.py" run --wt "$WT" --out "$OUT" --family "$FAMILY" --threads 8 --mem 30G "$@"; }
case $FAMILY in
  argsort)  sizes="16 64 256 1024"; a=("sort: grand-product permutation" gp); b=("sort: routed permutation" routed) ;;
  matmul)   sizes="8 16 32 64";     a=("linear: freivalds" freivalds); b=("linear: direct product sums" direct) ;;
  convolve) sizes="16 32 64 128 256"; a=("linear: polynomial identity" identity); b=("linear: direct product sums" direct) ;;
  *) echo "unknown family $FAMILY" >&2; exit 2 ;;
esac
for n in $sizes; do
  run --sizes "$n" --force "" --label auto "$@"
  run --sizes "$n" --force "${a[0]}" --label "${a[1]}" "$@"
  run --sizes "$n" --force "${b[0]}" --label "${b[1]}" "$@"
done
