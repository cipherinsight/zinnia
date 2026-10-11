#!/usr/bin/env bash
# The NPBench/Pythran scaling study's schedule, alongside the DS-1000 study (log logs/scaling.log):
#  1. setup of $RUN/zinnia-scnpb on CPUs 32-63 (untimed);
#  2. the CAP 32/64 count and validation jobs on CPUs 32-63 (queue npbu, SIBLINGS_ONLY), until the
#     DS-1000 timing queue (queue-scale) appears; it waits behind four gate jobs;
#  3. jobs left from 2, then a followup that appends CAP 32/64's timed jobs, then the CAP 128
#     untimed jobs and their followup, all appended to queue-scale after the DS-1000 shards; then
#     the gate opens (nothing of ours runs on CPUs 32-63 any more);
#  4. when queue-scale is done: the zkVMs alone on the whole machine, CAP 32 and 64, then 128.
RUN=/home/zhantong/zinnia-eval-run
S=$RUN/post-freeze/scripts
N=$RUN/zinnia-scnpb/benchmarking/scaling_npb/remote
O=$RUN/scaling-npb
J=$RUN/npb-data/jobs
mkdir -p $O
log() { echo "$* $(date -Is)"; }
bash $N/setup.sh 32-63 || { touch $O/gate-open; log "npb setup FAILED; gate opened for the DS-1000 timing"; exit 1; }
log "npb setup ok"
SIBLINGS_ONLY=1 STOP_FILE=$RUN/queue-scale/jobs bash $S/queue.sh npbu $J/jobs-untimed-32-64.txt
log "npbu stopped; $(grep -c . $RUN/queue-npbu/jobs) jobs left"
until [[ -f $RUN/queue-scale/jobs ]]; do sleep 60; done
{
  cat $RUN/queue-npbu/jobs
  echo "untimed bash $S/npb-followup.sh scale 32,64"
  cat $J/jobs-untimed-128.txt
  echo "untimed bash $S/npb-followup.sh scale 128"
} > $O/appended.txt
( flock 9; cat $O/appended.txt >> $RUN/queue-scale/jobs ) 9>$RUN/queue-scale/lock
touch $O/gate-open
log "appended $(wc -l < $O/appended.txt) jobs to queue-scale; gate open"
until grep -q "queue scale done" $RUN/logs/scaling.log; do sleep 120; done
bash $N/zkvm.sh P 32 64 > $O/zkvm-32-64.log 2>&1; log "zkvm P 32 64 exit $?"
bash $N/zkvm.sh 128 > $O/zkvm-128.log 2>&1; log "zkvm 128 exit $?"
log "npb driver done"
