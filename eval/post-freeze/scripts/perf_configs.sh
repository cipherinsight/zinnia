#!/usr/bin/env bash
# Perf-set costs (halo2-base cells, Noir gates) and IR statistics of one compiler
# configuration: the default build or one fact-layer ablation alone.
# Port of eval-redo/fix/fact-review-data/perf_configs.sh with the paths as arguments and
# the memory cap by capped.py (no systemd in the remote container).
#
# Usage: perf_configs.sh WT OUT CONFIG [NOIR_JOBS]
#   WT      Zinnia checkout (built; its halo2_hw baseline built in
#           benchmarking/baselines/halo2_hw/target/release)
#   OUT     output directory (costs_<c>.json, gates_<c>.json, ir_<c>.json and logs)
#   CONFIG  final | abl_smt | abl_layer | abl_check | abl_af | noir_hw (the baselines' Noir gates)
set -uo pipefail
WT=$1 OUT=$2 c=$3 JOBS=${4:-4}
S=$(cd "$(dirname "$0")" && pwd)
PY=$WT/.venv/bin/python
CAP="$PY $S/capped.py"
mkdir -p "$OUT"
cd "$WT"
export ZWT=$WT
case $c in
  final) env=() ;;
  abl_smt) env=(ZINNIA_ABLATE_SMT=1) ;;
  abl_layer) env=(ZINNIA_ABLATE_LAYER_FACTS=1) ;;
  abl_check) env=(ZINNIA_ABLATE_CHECK_ELIM=1) ;;
  abl_af) env=(ZINNIA_ABLATE_ARRAY_FACTS=1) ;;
  noir_hw)
    $PY benchmarking/noir_gates.py --baselines --jobs "$JOBS" --wrap "$CAP 8G 1200" \
      --out "$OUT/noir_hw.json" > "$OUT/noir_hw.log" 2>&1
    echo "noir_hw exit $?"; exit 0 ;;
  *) echo "unknown config $c" >&2; exit 2 ;;
esac
echo "== $c ${env[*]:-} $(date -Is)"
env "${env[@]}" $PY -m benchmarking.harness.costs_halo2 \
  --halo2-bin "$WT/benchmarking/baselines/halo2_hw/target/release/halo2_hw" \
  --out "$OUT/costs_$c.json" > "$OUT/costs_$c.log" 2>&1
echo "costs exit $? $(date -Is)"; tail -n 1 "$OUT/costs_$c.log"
env "${env[@]}" $PY benchmarking/noir_gates.py --zinnia --jobs "$JOBS" --wrap "$CAP 8G 1200" \
  --out "$OUT/gates_$c.json" > "$OUT/gates_$c.log" 2>&1
echo "gates exit $? $(date -Is)"
env "${env[@]}" $CAP 16G none $PY "$S/ir_stats.py" "$OUT/ir_$c.json" > "$OUT/ir_$c.log" 2>&1
echo "ir exit $? $(date -Is)"
