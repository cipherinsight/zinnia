#!/usr/bin/env bash
# After the caps' untimed jobs have all finished: merge the per-point counts, write each cap's
# per-step timeout ($OUT/timeout_cap<C>: 3 x the slowest compile + estimated prove, at least
# 14,400 s) and wall-time estimate ($OUT/estimate_cap<C>.json; also on stderr), and print the
# timed job lines (stage B) for queue.sh.
# Usage: timed-jobs.sh <caps, e.g. 32,64>  > $OUT/jobs-timed-<caps>.txt
source "$(dirname "$0")/env.sh"
cd $CO
for cap in ${1//,/ }; do
  for config in final fix_delegate; do
    parts=($OUT/counts/parts/cap$cap/$config/*.json)
    [[ -e ${parts[0]} ]] || { echo "no counts for cap $cap $config" >&2; exit 1; }
  done
  $PY -m benchmarking.scaling_npb merge-counts --cap $cap --out-dir $OUT/counts $OUT/counts/parts/cap$cap/*/*.json >&2
done
$PY -m benchmarking.scaling_npb jobs-timed --data $DATA --caps $1 --counts-dir $OUT/counts --script $JOB \
  --repeats ${REPEATS:-5} --heavy-repeats ${HEAVY_REPEATS:-3}
