# Paths of the NPBench/Pythran scaling study on the remote (sourced by the other scripts here).
RUN=${RUN:-/home/zhantong/zinnia-eval-run}
CO=${CO:-$RUN/zinnia-scnpb}                 # checkout of branch scaling-npb (src/ = 4115cecb)
DATA=${DATA:-$RUN/npb-data}                 # eval-redo/scaling/npb-data, put here by remote.sh
OUT=${OUT:-$RUN/scaling-npb}                # counts, validation, timeouts, estimates, logs, results
PY=$CO/.venv/bin/python
#: Untimed compiles (counts, validation) stop after 6 h: a resource limit.
COUNT_TIMEOUT=${COUNT_TIMEOUT:-21600}
JOB=$CO/benchmarking/scaling_npb/remote/job.sh
export ZINNIA_HARNESS_MEMORY_CAP=watchdog ZWT=$CO
ncpus() {  # number of CPUs in a taskset list such as 0-7,32-39
  local n=0 part
  for part in ${1//,/ }; do
    if [[ $part == *-* ]]; then n=$((n + ${part#*-} - ${part%-*} + 1)); else n=$((n + 1)); fi
  done
  echo $n
}
