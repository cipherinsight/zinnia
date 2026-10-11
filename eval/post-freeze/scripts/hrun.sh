#!/usr/bin/env bash
# One harness run (a shard of a session) for queue.sh: repeats 5, 8 threads, 30G, results in
# zinnia-src's results directory, log in $RUN/logs/<label>.log.
# Usage: hrun.sh <checkout> <label> <case set> <shard> <systems> [harness args...] <cpus>
RUN=/home/zhantong/zinnia-eval-run
RES=$RUN/zinnia-src/benchmarking/results
co=$1 label=$2 cs=$3 shard=$4 systems=$5; shift 5
cpus=${*: -1}; set -- "${@:1:$#-1}"
(cd $co && taskset -c $cpus .venv/bin/python -m benchmarking.harness run --label $label --case-set $cs \
  --shard $shard --systems $systems --repeats 5 --threads 8 --memory-max 30G --results-dir $RES "$@" \
  > $RUN/logs/$label.log 2>&1)
echo "$label exit $? $(date -Is) cpus $cpus"
