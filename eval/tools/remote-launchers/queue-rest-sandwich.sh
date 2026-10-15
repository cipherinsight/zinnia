#!/usr/bin/env bash
# Queue on CPUs 32-63 after the first paired zkVM epoch:
# (A) second paired hand-written session (Zinnia, halo2_hw, noir_hw, repeats 5);
# (B) NPBench/Pythran sandwich per shard: Zinnia x5, SP1 + RISC Zero x1, Zinnia x5 (same pinned CPUs);
# (D) DS-1000 zkVM sandwich re-run (the first one overlapped other CPU-heavy jobs): Zinnia x5, SP1 + RISC Zero +
#     Cairo x1, Zinnia x5;
# (C) NPB SP1 Groth16 x1.
cd /home/zhantong/zinnia-eval-run/zinnia-src
export ZINNIA_HARNESS_MEMORY_CAP=watchdog
DS=benchmarking/datasets/case_sets/ds1000_perf.json
NPB=benchmarking/datasets/case_sets/npbench_pythran_perf.json
run() {  # run <label> <cpus> <case set> <shard> <systems> <repeats> <cap> [extra...]
  local label=$1 cpus=$2 cs=$3 shard=$4 systems=$5 reps=$6 cap=$7; shift 7
  taskset -c $cpus .venv/bin/python -m benchmarking.harness run --label $label --case-set $cs --shard $shard \
    --systems $systems --repeats $reps --threads 8 --memory-max $cap "$@" > ../logs/$label.log 2>&1
  echo "$label exit $? $(date -Is)"
}
waves() {  # waves <function>: shards 0-3 then 4-7, 8 pinned CPUs each within 32-63
  local fn=$1
  for first in 0 4; do
    pids=()
    for j in 0 1 2 3; do $fn $((first + j)) $((32+8*j))-$((32+8*j+7)) & pids+=($!); done
    for p in "${pids[@]}"; do wait $p; done
  done
}
hw_shard() { run paired-r2-s$1 $2 $DS $1/8 zinnia_halo2,halo2_hw,zinnia_noir,noir_hw 5 30G; }
npb_shard() {
  local cap=30G; case $1 in 2|3) cap=64G;; esac
  run paired-npb-zpre-s$1 $2 $NPB $1/8 zinnia_halo2,zinnia_noir 5 $cap --timeout 14400
  run paired-npb-s$1 $2 $NPB $1/8 sp1,risc0 1 $cap --timeout 14400
  run paired-npb-zpost-s$1 $2 $NPB $1/8 zinnia_halo2,zinnia_noir 5 $cap --timeout 14400
}
ds_zkvm_shard() {
  run paired-zkvm2-zpre-s$1 $2 $DS $1/8 zinnia_halo2,zinnia_noir 5 30G
  run paired-zkvm2-s$1 $2 $DS $1/8 sp1,risc0,cairo 1 30G
  run paired-zkvm2-zpost-s$1 $2 $DS $1/8 zinnia_halo2,zinnia_noir 5 30G
}
echo "A start $(date -Is)"; waves hw_shard
echo "B start $(date -Is)"; waves npb_shard
echo "D start $(date -Is)"; waves ds_zkvm_shard
echo "C start $(date -Is)"
export PATH=$HOME/zinnia-eval-run/opt/r0g16/bin:$PATH
pids=()
for i in 0 1; do
  taskset -c $((32+16*i))-$((32+16*i+15)) .venv/bin/python -m benchmarking.harness run --label npb-sp1-groth16-r1-s$i \
    --case-set $NPB --shard $i/2 --systems sp1_groth16 --repeats 1 --threads 16 --memory-max 60G --timeout 14400 \
    > ../logs/npb-sp1-groth16-r1-s$i.log 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait $p; done
echo "queue done $(date -Is)"
