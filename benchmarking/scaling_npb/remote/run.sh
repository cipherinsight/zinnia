#!/usr/bin/env bash
# The NPBench/Pythran scaling study on the remote machine (64 logical CPUs: CPU c and 32+c are
# hyperthread siblings of one physical core; 251 GB RAM). Style of
# eval-redo/post-freeze/scripts/final-remeasure.sh and fin-queue.sh.
#
# Stages (default: S V P A B C, in this order; each waits for the previous one):
#   S  setup: the checkout's src/ must equal 4115cecb; point the case sets at $DATA; diff the
#      zkVM programs (zkvm-check); build the zkVM hosts and guests (sp1, risc0) if needed;
#      check-env. The Zinnia extension and noir-execute must already be built in $WT
#      (`harness build` / the copy from zinnia-post, as for every other session).
#   V  validation of the points this study could not validate locally ($DATA/validate_on_remote.txt,
#      lines "<cap> <case> <system>"): the scaling_npb validate check (mock outputs vs NumPy) with
#      the timing settings' memory cap and timeout; untimed, all CPUs  -> $OUT/validate/
#   P  thread probe: one short SP1 and one RISC Zero run on all 64 CPUs (--threads 64), watched
#      by thread_probe.py (threads per prover, busiest interval in CPUs) -> $OUT/probe/
#   A  counts (untimed, nothing timed runs): per cap, measure.py with configuration `final`
#      (halo2-base cells and Noir gates as Zinnia chooses) and `fix_delegate` (the same with the
#      backend's own array lowering), on all 64 CPUs               -> $OUT/counts/
#   B  Zinnia timing: zinnia_halo2, zinnia_halo2_delegate, zinnia_noir, zinnia_noir_delegate back to
#      back per entry in one harness run (the delegate systems are Zinnia with the backend's own
#      array lowering), every cap, 8 shards per cap, as a job queue over four slots of 8 physical
#      cores (CPUs 8j..8j+7; their siblings 32+8j.. stay idle), threads 8, memory cap 30G, repeats
#      $REPEATS_<cap> (5; never below 3, the same for all four systems at a cap), per-step timeout
#      $TIMEOUT. Only these four timed jobs run.                    -> $RES/<date>-scnpb-z-cap<C>-s<K>
#   C  zkVM timing: sp1 + risc0, one harness run per cap, one at a time on the whole machine
#      (taskset 0-63, threads 64, memory cap $ZKVM_MEMORY), repeats 1, the same per-step timeout.
#      Nothing else runs.                                            -> $RES/<date>-scnpb-k-cap<C>
#
# The asymmetry is deliberate (author's rule): Zinnia (all four systems) gets 8 cores and 30G
# exactly as in every other timing session; the zkVMs, far slower, get the whole machine. Same
# timeout for every system.
#
# Usage (on the remote, after `remote.sh push` of branch scaling-npb to $WT and `remote.sh put` of
# the data directory to $DATA):  bash $WT/benchmarking/scaling_npb/remote/run.sh [stages...]
set -uo pipefail
RUN=${RUN:-/home/zhantong/zinnia-eval-run}
WT=${WT:-$RUN/zinnia-post}                  # checkout of branch scaling-npb (src/ = 4115cecb), built
DATA=${DATA:-$RUN/scaling-npb-data}         # eval-redo/scaling/npb-data, copied
OUT=${OUT:-$RUN/scaling-npb}                # counts, probe, logs
RES=${RES:-$OUT/results}                    # harness epochs
ZK_SEED=${ZK_SEED:-$RUN/zinnia-src}         # a checkout whose sp1/risc0 target dirs seed the build
CAPS=${CAPS:-32 64 128}
TIMEOUT=${TIMEOUT:-14400}                   # per step, every system (as the NPB perf sessions)
REPEATS_32=${REPEATS_32:-5}
REPEATS_64=${REPEATS_64:-5}
REPEATS_128=${REPEATS_128:-5}
ZKVM_MEMORY=${ZKVM_MEMORY:-200G}
SHARDS=8
ZINNIA_SYSTEMS=zinnia_halo2,zinnia_halo2_delegate,zinnia_noir,zinnia_noir_delegate
PY=$WT/.venv/bin/python
export ZINNIA_HARNESS_MEMORY_CAP=watchdog ZWT=$WT
STAGES=${*:-S V P A B C}
mkdir -p $OUT/logs $OUT/counts $RES
cd $WT

case_set() { echo $DATA/case_sets/npbench_pythran_cap$1.json; }
repeats() { local v=REPEATS_$1; echo ${!v}; }
log() { echo "$* $(date -Is)"; }

setup() {
  git diff --quiet 4115cecb -- src || { echo "src/ differs from 4115cecb"; return 1; }
  # In the first NPB zkVM epoch SP1 died with SIGBUS on conv2d_bias while /dev/shm was Docker's
  # 64 MB default; the reruns with a larger /dev/shm passed. The scaled points are larger.
  echo "/dev/shm: $(df -h /dev/shm | tail -1)"
  $PY -m benchmarking.scaling_npb relocate --data $DATA
  $PY -m benchmarking.scaling_npb zkvm-check --data $DATA --out $OUT/zkvm_check.json | tail -1 || return 1
  for s in sp1 risc0; do
    t=benchmarking/baselines/$s/target
    [[ -d $t || ! -d $ZK_SEED/$t ]] || cp -r --reflink=auto $ZK_SEED/$t $t
  done
  taskset -c 0-63 $PY -m benchmarking.harness build --skip-zinnia --baselines sp1,risc0 --jobs 32 \
    > $OUT/logs/build-zkvm.log 2>&1 || { tail -5 $OUT/logs/build-zkvm.log; return 1; }
  $PY -m benchmarking.harness check-env --systems $ZINNIA_SYSTEMS,sp1,risc0 \
    --out $OUT/check-env.json > $OUT/logs/check-env.log 2>&1 || { tail -20 $OUT/logs/check-env.log; return 1; }
}

probe() {  # one small scaled point per zkVM, whole machine, as stage C runs it
  mkdir -p $OUT/probe
  for s in sp1 risc0; do
    $PY -m benchmarking.scaling_npb.thread_probe --out $OUT/probe/$s.json -- \
      taskset -c 0-63 $PY -m benchmarking.harness run --label scnpb-probe-$s --case-set $(case_set 32) \
      --cases npbench_polybench_gemm_gemm__kernel_cap32 --systems $s --repeats 1 --threads 64 \
      --memory-max $ZKVM_MEMORY --timeout $TIMEOUT --results-dir $OUT/probe/results > $OUT/logs/probe-$s.log 2>&1
    log "probe $s exit $?"
  done
}

validate_remote() {  # untimed: the points listed in $DATA/validate_on_remote.txt
  local list=$DATA/validate_on_remote.txt c system cases
  [[ -s $list ]] || { echo "nothing to validate on the remote"; return 0; }
  mkdir -p $OUT/validate
  for c in $CAPS; do
    for system in ${ZINNIA_SYSTEMS//,/ }; do
      cases=$(awk -v c=$c -v s=$system '$1 == c && $3 == s {print $2}' $list | paste -sd,)
      [[ -z $cases ]] && continue
      taskset -c 0-63 $PY -m benchmarking.scaling_npb validate --data $DATA --cap $c --cases $cases \
        --systems $system --jobs 6 --memory 30G --timeout $TIMEOUT \
        --out $OUT/validate/validate_cap${c}_$system.jsonl > $OUT/logs/validate-cap$c-$system.log 2>&1
      log "validate cap $c $system exit $?"
    done
  done
}

counts() {  # untimed: every cap, both configurations, all CPUs
  for c in $CAPS; do
    for config in final fix_delegate; do
      taskset -c 0-63 $PY benchmarking/ablations/measure.py --config $config --case-sets $(case_set $c) \
        --targets halo2,noir --gates --jobs 6 --timeout $((TIMEOUT + 100)) \
        --wrap "$PY benchmarking/ablations/capped.py 30G $TIMEOUT" \
        --out $OUT/counts/measure_${config}_cap$c.json > $OUT/logs/counts-${config}-cap$c.log 2>&1
      log "counts $config cap $c exit $?"
    done
  done
}

zinnia_job() {  # zinnia_job <cap> <shard> <cpus>
  local label=scnpb-z-cap$1-s$2
  taskset -c $3 $PY -m benchmarking.harness run --label $label --case-set $(case_set $1) \
    --shard $2/$SHARDS --systems $ZINNIA_SYSTEMS --repeats $(repeats $1) --threads 8 \
    --memory-max 30G --timeout $TIMEOUT --results-dir $RES > $OUT/logs/$label.log 2>&1
  log "$label exit $? cpus $3"
}

zinnia_queue() {  # four slots of 8 physical cores; a slot takes the next (cap, shard) when free
  local q=$OUT/zinnia-queue
  mkdir -p $q; : > $q/jobs
  for c in $CAPS; do for k in $(seq 0 $((SHARDS - 1))); do echo "$c $k" >> $q/jobs; done; done
  worker() {
    local cpus="$((8*$1))-$((8*$1+7))" line
    while true; do
      exec 9>$q/lock; flock 9
      line=$(head -n 1 $q/jobs); [[ -n $line ]] && sed -i 1d $q/jobs
      flock -u 9; exec 9>&-
      [[ -z $line ]] && return
      zinnia_job $line $cpus
    done
  }
  for j in 0 1 2 3; do worker $j & done
  wait
}

zkvm_runs() {  # one cap at a time, the whole machine
  for c in $CAPS; do
    local label=scnpb-k-cap$c
    taskset -c 0-63 $PY -m benchmarking.harness run --label $label --case-set $(case_set $c) \
      --systems sp1,risc0 --repeats 1 --threads 64 --memory-max $ZKVM_MEMORY --timeout $TIMEOUT \
      --results-dir $RES > $OUT/logs/$label.log 2>&1
    log "$label exit $?"
  done
}

log "scaling-npb start: commit $(git log -1 --format='%h %s'), stages $STAGES"
for s in $STAGES; do
  log "$s start"
  case $s in
    S) setup || { log "setup failed"; exit 1; } ;;
    V) validate_remote ;;
    P) probe ;;
    A) counts ;;
    B) zinnia_queue ;;
    C) zkvm_runs ;;
    *) echo "unknown stage $s"; exit 2 ;;
  esac
  log "$s done"
done
log "all done"
