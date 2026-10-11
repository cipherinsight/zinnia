#!/usr/bin/env bash
# A job queue over four slots of the remote's 32 physical cores (the logic of fin-queue.sh,
# for any job list). Slot j = physical cores 8j..8j+7; CPU 32+c is the hyperthread sibling of c.
# Job file: one job per line, "timed <command...>" or "untimed <command...>"; the slot's CPU
# list is appended to the command. A timed job gets the slot's 8 physical cores. An untimed job
# gets the same while any timed job runs, and all 16 logical CPUs of its slot once none does.
# Nothing ever runs on a timed job's sibling CPUs.
# A worker that finds the queue empty waits while another worker is still busy (a running job may
# append more lines, under $Q/lock), and stops when every worker is idle.
# Environment:
#   SIBLINGS_ONLY=1  slot j is CPUs 32+8j..32+8j+7 only, untimed jobs only (for count work while
#                    something untimed of ours holds CPUs 0-31)
#   STOP_FILE=path   a worker takes no new job once this file exists (the jobs left stay in
#                    $Q/jobs for the caller to move elsewhere)
# Usage: queue.sh <name> <job file>     (state in $RUN/queue-<name>/; append to its jobs under its lock)
RUN=/home/zhantong/zinnia-eval-run
export ZINNIA_HARNESS_MEMORY_CAP=watchdog
Q=$RUN/queue-$1
mkdir -p $Q/timed $Q/busy
rm -f $Q/timed/* $Q/busy/*
cp $2 $Q/jobs

pop() {  # print and remove the first queued job (empty when the queue is empty)
  local line
  exec 9>$Q/lock; flock 9
  line=$(head -n 1 $Q/jobs)
  [[ -n $line ]] && sed -i 1d $Q/jobs
  flock -u 9; exec 9>&-
  printf '%s' "$line"
}
worker() {  # worker <slot>
  local j=$1 phys all line kind cpus
  if [[ ${SIBLINGS_ONLY:-} == 1 ]]; then
    phys="$((32+8*$1))-$((32+8*$1+7))"; all=$phys
  else
    phys="$((8*$1))-$((8*$1+7))"; all="$((8*$1))-$((8*$1+7)),$((32+8*$1))-$((32+8*$1+7))"
  fi
  while true; do
    [[ -n ${STOP_FILE:-} && -e $STOP_FILE ]] && break
    touch $Q/busy/slot$j
    line=$(pop)
    if [[ -z $line ]]; then
      rm -f $Q/busy/slot$j
      [[ -z $(ls $Q/busy) ]] && break
      sleep 30; continue
    fi
    set -- $line; kind=$1; shift
    if [[ $kind == timed ]]; then
      touch $Q/timed/slot$j; "$@" $phys; rm -f $Q/timed/slot$j
    else
      cpus=$phys
      [[ -z $(ls $Q/timed) ]] && cpus=$all
      "$@" $cpus
    fi
  done
  rm -f $Q/busy/slot$j
}
echo "queue $1 start $(date -Is)"
for j in 0 1 2 3; do worker $j & done
wait
echo "queue $1 done $(date -Is)"
