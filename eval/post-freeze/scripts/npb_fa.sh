#!/usr/bin/env bash
# Fact-layer ablation of the NPBench/Pythran perf set (perf_configs.sh covers DS-1000 only):
# halo2-base cells, Noir gates and digests per case under one configuration, by measure.py.
# The reference (final) is measure_final.json of run_configs.sh, which covers both case sets.
# Usage: npb_fa.sh <label> <NAME=VALUE> <cpus>      (output $D/perf-npb/measure_<label>.json)
RUN=/home/zhantong/zinnia-eval-run
POST=$RUN/zinnia-post
OUT=$RUN/post-freeze/data-fin/perf-npb
label=$1 env=$2 cpus=$3
n=0; for part in ${cpus//,/ }; do [[ $part == *-* ]] && n=$((n + ${part#*-} - ${part%-*} + 1)) || n=$((n + 1)); done
mkdir -p $OUT
cd $POST
taskset -c $cpus .venv/bin/python benchmarking/ablations/measure.py --config $label --env $env \
  --case-sets npbench_pythran --gates --jobs $((n / 2)) --timeout 3700 \
  --wrap ".venv/bin/python benchmarking/ablations/capped.py 24G 3600" \
  --out $OUT/measure_$label.json > $OUT/measure_$label.log 2>&1
echo "npb-fa $label exit $? $(date -Is) cpus $cpus"
