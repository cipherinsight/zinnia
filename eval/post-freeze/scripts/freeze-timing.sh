#!/usr/bin/env bash
# Timing at the new freeze (array-ssa-fix), all on CPUs 0-31 (4 shards x 8 pinned cores), nothing else of ours on
# the machine. Zinnia and the hand-written baselines run from $RUN/zinnia-post (new freeze); the zkVMs from
# $RUN/zinnia-src (compiler-independent builds). All epochs go to zinnia-src/benchmarking/results.
#   H1  paired hand-written session (zinnia_halo2, halo2_hw, zinnia_noir, noir_hw; repeats 5)
#   Z   DS-1000 zkVM sandwich per shard: Zinnia x5, SP1 + RISC Zero + Cairo x1, Zinnia x5
#   H2  paired hand-written session
#   N   NPBench/Pythran sandwich per shard: Zinnia x5, SP1 + RISC Zero x1, Zinnia x5
#   H3  paired hand-written session
# Usage: freeze-timing.sh <tag> [stages...]   (default stages: H1 Z H2 N H3)
RUN=/home/zhantong/zinnia-eval-run
POST=$RUN/zinnia-post
SRC=$RUN/zinnia-src
RES=$SRC/benchmarking/results
export ZINNIA_HARNESS_MEMORY_CAP=watchdog
DS=benchmarking/datasets/case_sets/ds1000_perf.json
NPB=benchmarking/datasets/case_sets/npbench_pythran_perf.json
TAG=$1; shift
STAGES=${*:-H1 Z H2 N H3}
run() {  # run <checkout> <label> <cpus> <case set> <shard> <systems> <repeats> <cap> [extra...]
  local wt=$1 label=$2 cpus=$3 cs=$4 shard=$5 systems=$6 reps=$7 cap=$8; shift 8
  (cd $wt && taskset -c $cpus .venv/bin/python -m benchmarking.harness run --label $label --case-set $cs \
    --shard $shard --systems $systems --repeats $reps --threads 8 --memory-max $cap --results-dir $RES "$@" \
    > $RUN/logs/$label.log 2>&1)
  echo "$label exit $? $(date -Is)"
}
waves() {  # waves <function> <stage>: shards 0-3 then 4-7, 8 pinned cores each within 0-31
  local fn=$1 stage=$2
  for first in 0 4; do
    pids=()
    for j in 0 1 2 3; do $fn $((first + j)) $((8*j))-$((8*j+7)) $stage & pids+=($!); done
    for p in "${pids[@]}"; do wait $p; done
  done
}
hw_shard() { run $POST $TAG-$3-s$1 $2 $DS $1/8 zinnia_halo2,halo2_hw,zinnia_noir,noir_hw 5 30G; }
ds_zkvm_shard() {
  run $POST $TAG-zpre-s$1 $2 $DS $1/8 zinnia_halo2,zinnia_noir 5 30G
  run $SRC $TAG-zkvm-s$1 $2 $DS $1/8 sp1,risc0,cairo 1 30G
  run $POST $TAG-zpost-s$1 $2 $DS $1/8 zinnia_halo2,zinnia_noir 5 30G
}
npb_shard() {
  local cap=30G; case $1 in 2|3) cap=64G;; esac
  run $POST $TAG-npb-zpre-s$1 $2 $NPB $1/8 zinnia_halo2,zinnia_noir 5 $cap --timeout 14400
  run $SRC $TAG-npb-zkvm-s$1 $2 $NPB $1/8 sp1,risc0 1 $cap --timeout 14400
  run $POST $TAG-npb-zpost-s$1 $2 $NPB $1/8 zinnia_halo2,zinnia_noir 5 $cap --timeout 14400
}
echo "commit $(git -C $POST log -1 --oneline)"
for s in $STAGES; do
  echo "$s start $(date -Is)"
  case $s in
    H*) waves hw_shard ${s,,} ;;
    Z) waves ds_zkvm_shard z ;;
    N) waves npb_shard n ;;
  esac
done
echo "all done $(date -Is)"
