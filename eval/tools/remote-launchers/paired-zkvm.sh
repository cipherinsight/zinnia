#!/usr/bin/env bash
# Paired zkVM epoch on DS-1000. Per shard, on the same 8 pinned CPUs and back to back: Zinnia (repeats 5), then the
# zkVMs (repeats 1; they are slow). 8 shards in two waves of 4 on CPUs 32-63, threads 8, 30G.
cd /home/zhantong/zinnia-eval-run/zinnia-src
export ZINNIA_HARNESS_MEMORY_CAP=watchdog
CS=benchmarking/datasets/case_sets/ds1000_perf.json
shard() {  # shard <i> <cpus>
  taskset -c $2 .venv/bin/python -m benchmarking.harness run --label paired-zkvm-z-s$1 --case-set $CS --shard $1/8 \
    --systems zinnia_halo2,zinnia_noir --repeats 5 --threads 8 --memory-max 30G > ../logs/paired-zkvm-z-s$1.log 2>&1
  echo "shard $1 zinnia exit $?"
  taskset -c $2 .venv/bin/python -m benchmarking.harness run --label paired-zkvm-s$1 --case-set $CS --shard $1/8 \
    --systems sp1,risc0,cairo --repeats 1 --threads 8 --memory-max 30G > ../logs/paired-zkvm-s$1.log 2>&1
  echo "shard $1 zkvm exit $?"
}
for first in 0 4; do
  pids=()
  for j in 0 1 2 3; do shard $((first + j)) $((32+8*j))-$((32+8*j+7)) & pids+=($!); done
  for p in "${pids[@]}"; do wait $p; done
done
echo "all done $(date -Is)"
bash /home/zhantong/zinnia-eval-run/queue-rest.sh
