#!/usr/bin/env bash
# Print the untimed job lines (stages V and A) of the given caps for queue.sh.
# Usage: untimed-jobs.sh <caps, e.g. 32,64>  > $OUT/jobs-untimed-<caps>.txt
source "$(dirname "$0")/env.sh"
cd $CO && $PY -m benchmarking.scaling_npb jobs-untimed --data $DATA --caps $1 --script $JOB
