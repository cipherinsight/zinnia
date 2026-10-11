#!/usr/bin/env bash
# A queue job placed after a set of NPBench/Pythran untimed jobs: waits until none of the given
# caps' count/validate jobs still runs, then writes those caps' timed jobs (timed-jobs.sh: counts
# merged, per-cap timeout) and appends them to the same queue under its lock.
# Usage (job line): untimed bash npb-followup.sh <queue name> <caps, e.g. 32,64>   (+ CPU list, ignored)
RUN=/home/zhantong/zinnia-eval-run
N=$RUN/zinnia-scnpb/benchmarking/scaling_npb/remote
O=$RUN/scaling-npb
Q=$RUN/queue-$1 caps=$2
pat="job.sh (count|validate) (${caps//,/|}) "
while pgrep -f -- "$pat" > /dev/null; do sleep 60; done
if bash $N/timed-jobs.sh $caps > $O/jobs-timed-${caps//,/-}.txt 2> $O/timed-jobs-${caps//,/-}.log; then
  ( flock 9; cat $O/jobs-timed-${caps//,/-}.txt >> $Q/jobs ) 9>$Q/lock
  echo "npb-followup $caps: appended $(wc -l < $O/jobs-timed-${caps//,/-}.txt) timed jobs $(date -Is)"
else
  echo "npb-followup $caps: timed-jobs.sh FAILED (see $O/timed-jobs-${caps//,/-}.log) $(date -Is)"
fi
