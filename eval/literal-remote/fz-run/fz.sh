#!/usr/bin/env bash
# Final RQ1/RQ2 re-measurement on one compiler (epoch fz, checkout zinnia-sizefix at the commit
# below). Builds the compiler and both halo2 baselines from that commit and refuses a stale build.
# Timed phase, all 64 CPUs at once (8 pinned cores and 8 threads per job):
#   all 214 RQ2 programs x six systems (Zinnia, hand-written, direct translation, on halo2-base
#   and Noir), 3 proofs each, all systems of a program in one session: DS-1000 3 shards on
#   CPUs 0-23, NPBench/Pythran 3 shards on CPUs 24-47, 28G each; cases that hit the cap rerun
#   alone at 200G afterwards;
#   the Cairo sandwich of the 10 multi-output NPBench/Pythran programs (epoch cz10b) on CPUs 48-63.
# Untimed phase, all at once afterwards: compile timing (CPUs 0-15), acceptance of the 327
# programs (CPUs 16-31), the 214-program size ablations (CPUs 32-63).
RUN=/home/zhantong/zinnia-eval-run
WT=$RUN/zinnia-sizefix
PF=$RUN/post-freeze
O=$RUN/fz
RES=$O/epochs
export ZINNIA_HARNESS_MEMORY_CAP=watchdog
SYS=zinnia_halo2,halo2_hw,halo2_lit,zinnia_noir,noir_hw,noir_lit
echo "start $(date -Is)  commit $(git -C $WT log -1 --oneline)"
(cd $WT && .venv/bin/python -m benchmarking.harness build --jobs 64 --baselines halo2_hw,halo2_lit \
  > $O/build.log 2>&1); echo "build exit $? $(date -Is)"
built=$(cd $WT && .venv/bin/python -c "import json;print(json.load(open('target/zinnia-harness-build.json'))['source']['commit'])")
head=$(git -C $WT rev-parse HEAD)
[ "$built" = "$head" ] || { echo "STALE BUILD: record $built, HEAD $head"; exit 1; }
run() {  # run <label> <cpus> <case set> <shard|-> <cap> [extra...]
  local label=$1 cpus=$2 cs=$3 shard=$4 cap=$5; shift 5
  local sh=(); [ "$shard" != - ] && sh=(--shard "$shard")
  (cd $WT && taskset -c $cpus .venv/bin/python -m benchmarking.harness run --label $label --case-set $cs \
    "${sh[@]}" --systems $SYS --repeats 3 --threads 8 --memory-max $cap --results-dir $RES "$@" \
    > $O/$label.log 2>&1)
  echo "$label exit $? $(date -Is)"
}
(cd $WT && taskset -c 0-7 .venv/bin/python -m benchmarking.harness run --label fz-warm --case-set $O/fz-ds.json \
  --cases ds1000_case0453 --systems $SYS --repeats 1 --threads 8 --memory-max 28G --results-dir $O/warm \
  > $O/fz-warm.log 2>&1); echo "warm exit $? $(date -Is)"
pids=()
for j in 0 1 2; do run fz-ds-s$j $((8*j))-$((8*j+7)) $O/fz-ds.json $j/3 28G & pids+=($!); done
for j in 0 1 2; do run fz-npb-s$j $((24+8*j))-$((24+8*j+7)) $O/fz-npb.json $j/3 28G --timeout 14400 & pids+=($!); done
(cd $RUN/cz78 && RUN=$RUN CZ_TAG=cz10b CZ_CHECKOUT=zinnia-sizefix CZ_GROUPS=48,56 CZ_BUDGET_GB=60 \
  $WT/.venv/bin/python cz78-run.py $(cat $O/cz10-cases.txt) > $O/cz10b.log 2>&1; echo "cz10b exit $? $(date -Is)") & pids+=($!)
for p in "${pids[@]}"; do wait $p; done
big=$(cd $WT && .venv/bin/python - "$RES" <<'PY'
import json, sys, pathlib
cases = set()
for rec in pathlib.Path(sys.argv[1]).glob("*/records/*/*.json"):
    r = json.loads(rec.read_text())
    if r.get("status") in ("out_of_memory", "timeout"):
        cases.add(r["case"])
print(",".join(sorted(cases)))
PY
)
echo "B cases: ${big:-none}"
ds_big=$(echo "$big" | tr , '\n' | grep '^ds1000_' | paste -sd,)
npb_big=$(echo "$big" | tr , '\n' | grep -v '^ds1000_' | grep . | paste -sd,)
[ -n "$ds_big" ] && run fz-big-ds 0-7 $O/fz-ds.json - 200G --cases "$ds_big" --timeout 43200
[ -n "$npb_big" ] && run fz-big-npb 0-7 $O/fz-npb.json - 200G --cases "$npb_big" --timeout 43200
echo "TIMED_DONE $(date -Is)"
(cd $WT && .venv/bin/python $PF/scripts/compile_timing_remote.py --repo . --out $O/ct.jsonl \
   --jobs 16 --cpus 0-15 --timeout 300 --mem 8G > $O/ct.log 2>&1; echo "CT exit $? $(date -Is)") &
(cd $WT && .venv/bin/python $PF/scripts/accept_sharded.py $O/accept 16-31 > $O/accept.log 2>&1
 echo "ACC exit $? $(date -Is)") &
(cd $RUN/abl214 && taskset -c 32-63 $WT/.venv/bin/python run.py --checkout $WT --out $O/abl214-out \
   --accepted $RUN/abl214/rq1-ds-accepted.txt --jobs 32 --mem-cap 64G --min-free 40G --timeout 3600 \
   > $O/abl.log 2>&1 \
 && $WT/.venv/bin/python analyze.py $O/abl214-out --json $O/abl214-out/summary.json \
   > $O/abl214-out/tables.md; echo "ABL exit $? $(date -Is)") &
wait
echo "FZ_DONE $(date -Is)"
