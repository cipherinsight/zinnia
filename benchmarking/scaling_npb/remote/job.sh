#!/usr/bin/env bash
# One job of the scaling study for queue.sh; the queue appends the slot's CPU list.
#   job.sh count    <cap> <case> <final|fix_delegate> <cpus>   untimed: halo2 cells + Noir gates of one
#                       point under one configuration (ablations/measure.py, 30G, $COUNT_TIMEOUT
#                       per target)                         -> $OUT/counts/parts/cap<C>/<config>/<case>.json
#   job.sh validate <cap> <case> <system> <cpus>              untimed: the output check of one Zinnia
#                       system at one point (30G, $COUNT_TIMEOUT)
#                                                           -> $OUT/validate/cap<C>__<case>__<system>.jsonl
#   job.sh time     <cap> <case> <repeats> <delegates|-> <cpus>   timed: zinnia_halo2 + zinnia_noir at
#                       one point, then each listed delegate system, back to back in the slot: 8
#                       threads, 30G, the cap's per-step timeout ($OUT/timeout_cap<C>)
#                                                           -> $OUT/results/<date>-scnpb-z-<case>[-<system>]
# Untimed jobs set the thread variables to the number of CPUs they get. Log: $OUT/logs/<job>.log.
source "$(dirname "$0")/env.sh"
cpus=${*: -1}; set -- "${@:1:$#-1}"
kind=$1 cap=$2 case=$3
cd $CO
untimed_env() { local n; n=$(ncpus $cpus); export RAYON_NUM_THREADS=$n OMP_NUM_THREADS=$n HARDWARE_CONCURRENCY=$n; }
harness() {  # harness <label> <systems> <repeats>
  taskset -c $cpus $PY -m benchmarking.harness run --label $1 --case-set $DATA/case_sets/npbench_pythran_cap$cap.json \
    --cases $case --systems $2 --repeats $3 --threads 8 --memory-max 30G --timeout $(cat $OUT/timeout_cap$cap) \
    --results-dir $OUT/results > $OUT/logs/$1.log 2>&1
  echo "$1 exit $? $(date -Is) cpus $cpus"
}
case $kind in
  count)
    config=$4 name=count-$4-$case
    untimed_env
    mkdir -p $OUT/counts/parts/cap$cap/$config
    taskset -c $cpus $PY benchmarking/ablations/measure.py --config $config \
      --case-sets $DATA/case_sets/npbench_pythran_cap$cap.json --cases $case --targets halo2,noir --gates --jobs 1 \
      --timeout $((COUNT_TIMEOUT + 100)) --wrap "$PY benchmarking/ablations/capped.py 30G $COUNT_TIMEOUT" \
      --out $OUT/counts/parts/cap$cap/$config/$case.json > $OUT/logs/$name.log 2>&1
    echo "$name exit $? $(date -Is) cpus $cpus" ;;
  validate)
    system=$4 name=validate-$4-$case
    untimed_env
    taskset -c $cpus $PY -m benchmarking.scaling_npb validate --data $DATA --cap $cap --cases $case --systems $system \
      --jobs 1 --memory 30G --timeout $COUNT_TIMEOUT --out $OUT/validate/cap${cap}__${case}__$system.jsonl \
      > $OUT/logs/$name.log 2>&1
    echo "$name exit $? $(date -Is) cpus $cpus" ;;
  time)
    repeats=$4 delegates=$5
    [[ -s $OUT/timeout_cap$cap ]] || { echo "time $case: no $OUT/timeout_cap$cap (run timed-jobs.sh after the counts)"; exit 1; }
    harness scnpb-z-$case zinnia_halo2,zinnia_noir $repeats
    [[ $delegates == - ]] || for system in ${delegates//,/ }; do harness scnpb-z-$case-$system $system $repeats; done ;;
  *) echo "unknown job $kind"; exit 2 ;;
esac
