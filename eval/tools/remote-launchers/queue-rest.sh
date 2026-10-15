#!/usr/bin/env bash
# After the first paired zkVM epoch: only the compiler-independent job, NPB SP1 Groth16 x1 on CPUs 32-63.
# (Zinnia-side timing waits for the re-freeze on array-ssa; see queue-rest-sandwich.sh for that design.)
cd /home/zhantong/zinnia-eval-run/zinnia-src
export ZINNIA_HARNESS_MEMORY_CAP=watchdog
NPB=benchmarking/datasets/case_sets/npbench_pythran_perf.json
echo "C start $(date -Is)"
pids=()
for i in 0 1; do
  taskset -c $((32+16*i))-$((32+16*i+15)) .venv/bin/python -m benchmarking.harness run --label npb-sp1-groth16-r1-s$i \
    --case-set $NPB --shard $i/2 --systems sp1_groth16 --repeats 1 --threads 16 --memory-max 60G --timeout 14400 \
    > ../logs/npb-sp1-groth16-r1-s$i.log 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait $p; done
echo "queue done $(date -Is)"
