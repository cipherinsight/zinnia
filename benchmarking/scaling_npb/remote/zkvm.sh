#!/usr/bin/env bash
# The zkVM part of the scaling study, run after the queue is empty: nothing else may run.
#   P  thread probe: one short SP1 and one RISC Zero harness run (gemm, CAP 32) on all 64 CPUs with
#      --threads 64, watched by thread_probe.py (threads per prover, busiest interval in CPUs)
#                                                                       -> $OUT/probe/
#   C  per cap in the order given: sp1 + risc0 on every scaled point, one harness run on the whole
#      machine (taskset 0-63, --threads 64, memory cap $ZKVM_MEMORY), repeats 1, the cap's per-step
#      timeout ($OUT/timeout_cap<C>, the same as Zinnia's)               -> $OUT/results/<date>-scnpb-k-cap<C>
# The asymmetry is deliberate (author's rule): Zinnia ran on 8 pinned cores with 30G; the zkVMs,
# far slower, get the whole machine.
# Usage: zkvm.sh [P] [32] [64] [128]      (default: P 32 64 128)
source "$(dirname "$0")/env.sh"
ZKVM_MEMORY=${ZKVM_MEMORY:-200G}
cd $CO
for step in ${*:-P 32 64 128}; do
  if [[ $step == P ]]; then
    mkdir -p $OUT/probe
    for s in sp1 risc0; do
      $PY -m benchmarking.scaling_npb.thread_probe --out $OUT/probe/$s.json -- \
        taskset -c 0-63 $PY -m benchmarking.harness run --label scnpb-probe-$s \
        --case-set $DATA/case_sets/npbench_pythran_cap32.json --cases npbench_polybench_gemm_gemm__kernel_cap32 \
        --systems $s --repeats 1 --threads 64 --memory-max $ZKVM_MEMORY --timeout 14400 \
        --results-dir $OUT/probe/results > $OUT/logs/probe-$s.log 2>&1
      echo "probe $s exit $? $(date -Is)"
    done
    continue
  fi
  [[ -s $OUT/timeout_cap$step ]] || { echo "cap $step: no $OUT/timeout_cap$step (counts first)"; exit 1; }
  label=scnpb-k-cap$step
  taskset -c 0-63 $PY -m benchmarking.harness run --label $label --case-set $DATA/case_sets/npbench_pythran_cap$step.json \
    --systems sp1,risc0 --repeats 1 --threads 64 --memory-max $ZKVM_MEMORY --timeout $(cat $OUT/timeout_cap$step) \
    --results-dir $OUT/results > $OUT/logs/$label.log 2>&1
  echo "$label exit $? $(date -Is)"
done
