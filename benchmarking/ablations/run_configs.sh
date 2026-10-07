#!/usr/bin/env bash
# Ablation measurements of the array-lowering layer (docs/ablations.md): one compiler
# configuration per call. A copy of eval-redo/post-freeze/scripts/perf_configs.sh with the
# layer's configurations; runs unattended, every compile in a child under a memory cap
# (capped.py; no systemd needed).
#
# Usage: run_configs.sh WT OUT CONFIG [JOBS]
#   WT      Zinnia checkout (built: .venv with the extension; for the DS-1000 halo2-base
#           comparison also benchmarking/baselines/halo2_hw/target/release/halo2_hw)
#   OUT     output directory
#   CONFIG  final         the compiler (reference for every other configuration)
#           abl_joint     ZINNIA_ABLATE_JOINT=1: each operator group chooses alone
#           fix_delegate  ZINNIA_FORCE_CONSTRUCTION=@delegate: runtime-index accesses through
#                         the backend's memory (Noir native ROM/RAM; halo2-base table/RAM trace)
#           fix_selects   ZINNIA_FORCE_CONSTRUCTION=@selects: select chains / plain compute
#           oracle        every applicable construction forced, one family at a time
#           noir_hw       the handwritten Noir baselines' gates (once, for the comparison)
#   JOBS    parallel children (default 8)
#
# Outputs per configuration C (not oracle / noir_hw):
#   costs_C.json   DS-1000 perf: Zinnia vs halo2_hw cells at the common lookup width
#   gates_C.json   DS-1000 perf: Zinnia Noir bb gates (compare with noir_hw.json:
#                  benchmarking/noir_gates.py --compare gates_C.json noir_hw.json)
#   measure_C.json DS-1000 and NPBench/Pythran perf: cells, gates, IR/main.nr digests and
#                  the layer's decisions per group (fallbacks, native ROM/RAM groups)
set -uo pipefail
WT=$1 OUT=$2 c=$3 JOBS=${4:-8}
S=$WT/benchmarking/ablations
PY=$WT/.venv/bin/python
CAP="$PY $S/capped.py"
mkdir -p "$OUT"
cd "$WT"
export ZWT=$WT
case $c in
  final) env=() ;;
  abl_joint) env=(ZINNIA_ABLATE_JOINT=1) ;;
  fix_delegate) env=(ZINNIA_FORCE_CONSTRUCTION=@delegate) ;;
  fix_selects) env=(ZINNIA_FORCE_CONSTRUCTION=@selects) ;;
  oracle)
    echo "== oracle $(date -Is)"
    $PY "$S/oracle.py" --jobs "$JOBS" --timeout 3700 --wrap "$CAP 24G 3600" --out "$OUT/oracle.json" > "$OUT/oracle.log" 2>&1
    echo "oracle exit $? $(date -Is)"; tail -n 1 "$OUT/oracle.log"; exit 0 ;;
  noir_hw)
    $PY benchmarking/noir_gates.py --baselines --jobs "$JOBS" --timeout 1300 --wrap "$CAP 8G 1200" \
      --out "$OUT/noir_hw.json" > "$OUT/noir_hw.log" 2>&1
    echo "noir_hw exit $?"; exit 0 ;;
  *) echo "unknown config $c" >&2; exit 2 ;;
esac
echo "== $c ${env[*]:-} $(date -Is)"
env "${env[@]}" $PY -m benchmarking.harness.costs_halo2 \
  --halo2-bin "$WT/benchmarking/baselines/halo2_hw/target/release/halo2_hw" \
  --out "$OUT/costs_$c.json" > "$OUT/costs_$c.log" 2>&1
echo "costs exit $? $(date -Is)"; tail -n 1 "$OUT/costs_$c.log"
env "${env[@]}" $PY benchmarking/noir_gates.py --zinnia --jobs "$JOBS" --timeout 3700 --wrap "$CAP 24G 3600" \
  --out "$OUT/gates_$c.json" > "$OUT/gates_$c.log" 2>&1
echo "gates exit $? $(date -Is)"
$PY "$S/measure.py" --config "$c" --gates --timeout 3700 --jobs "$JOBS" --wrap "$CAP 24G 3600" \
  --out "$OUT/measure_$c.json" > "$OUT/measure_$c.log" 2>&1
echo "measure exit $? $(date -Is)"; tail -n 1 "$OUT/measure_$c.log"
