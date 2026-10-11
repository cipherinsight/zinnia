#!/usr/bin/env bash
# Timed proving + verification: Zinnia, the hand-optimized and the literal baselines on both
# backends (SYS overrides), 3 repeats, all systems of a case in the same session (8 pinned cores,
# --threads 8).
#   D  DS-1000: 8 shards in two waves of 4 on CPUs 0-31, 30G each
#   N  NPBench/Pythran: 8 shards in waves of 2 on CPUs 0-15, 64G each (at most two 64G jobs)
#   B  cases that hit the cap, one at a time on CPUs 0-7 with 200G (reported separately)
# Usage: three-way.sh <tag> [stages...]   (default: D N B)
RUN=/home/zhantong/zinnia-eval-run
WT=$RUN/zinnia-literal
RES=$WT/benchmarking/results
export ZINNIA_HARNESS_MEMORY_CAP=watchdog
DS=benchmarking/datasets/case_sets/ds1000_perf.json
NPB=benchmarking/datasets/case_sets/npbench_pythran_perf.json
SYS=${SYS:-zinnia_halo2,halo2_hw,halo2_lit,zinnia_noir,noir_hw,noir_lit}
TAG=$1; shift
STAGES=${*:-D N B}
run() {  # run <label> <cpus> <case set> <shard|-> <cap> [extra...]
  local label=$1 cpus=$2 cs=$3 shard=$4 cap=$5; shift 5
  local sh=(); [ "$shard" != - ] && sh=(--shard "$shard")
  (cd $WT && taskset -c $cpus .venv/bin/python -m benchmarking.harness run --label $label --case-set $cs \
    "${sh[@]}" --systems $SYS --repeats 3 --threads 8 --memory-max $cap --results-dir $RES "$@" \
    > $RUN/logs/$label.log 2>&1)
  echo "$label exit $? $(date -Is)"
}
echo "commit $(git -C $WT log -1 --oneline)  load $(cut -d' ' -f1-3 /proc/loadavg)"
for s in $STAGES; do
  echo "$s start $(date -Is)"
  case $s in
    D) for first in 0 4; do pids=()
         for j in 0 1 2 3; do run $TAG-ds-s$((first+j)) $((8*j))-$((8*j+7)) $DS $((first+j))/8 30G & pids+=($!); done
         for p in "${pids[@]}"; do wait $p; done; done ;;
    N) for first in 0 2 4 6; do pids=()
         for j in 0 1; do run $TAG-npb-s$((first+j)) $((8*j))-$((8*j+7)) $NPB $((first+j))/8 64G --timeout 14400 & pids+=($!); done
         for p in "${pids[@]}"; do wait $p; done; done ;;
    B) big=$(cd $WT && .venv/bin/python - "$RES" "$TAG" <<'PY'
import json, sys, pathlib
res, tag = pathlib.Path(sys.argv[1]), sys.argv[2]
cases = set()
for rec in res.glob(f"*-{tag}-*/records/*/*.json"):
    r = json.loads(rec.read_text())
    if r.get("status") in ("out_of_memory", "timeout"):
        cases.add(r["case"])
print(",".join(sorted(cases)))
PY
)
       echo "B cases: ${big:-none}"
       ds_big=$(echo "$big" | tr , '\n' | grep '^ds1000_' | paste -sd,)
       npb_big=$(echo "$big" | tr , '\n' | grep -v '^ds1000_' | paste -sd,)
       [ -n "$ds_big" ] && run $TAG-big-ds 0-7 $DS - 200G --cases "$ds_big" --timeout 43200
       [ -n "$npb_big" ] && run $TAG-big-npb 0-7 $NPB - 200G --cases "$npb_big" --timeout 43200
       ;;
  esac
done
echo "all done $(date -Is)"
