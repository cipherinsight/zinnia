#!/usr/bin/env bash
# The rest of final-remeasure.sh as a job queue over four slots, so no slot idles behind a long shard.
# Slot j = physical cores 8j..8j+7 (CPUs 0-31 are the 32 physical cores; CPU 32+c is the hyperthread
# sibling of CPU c). Timed jobs (harness sessions) get the slot's 8 physical cores, as in every
# earlier session. Count-only jobs (circuit size, ablations, acceptance, micro sweeps) get the same
# 8 cores while any timed job runs anywhere, and all 16 logical CPUs of their slot once none is left
# (every timed job is queued before every count-only job). Compile time and scaling run last, alone,
# exactly as in final-remeasure.sh.
# Usage: fin-queue.sh <pid0> <pid1> <pid2> <pid3>   (a still-running process per slot to wait for, or -)
RUN=/home/zhantong/zinnia-eval-run
POST=$RUN/zinnia-post
PF=$RUN/post-freeze
SC=$RUN/zinnia-scale
RES=$RUN/zinnia-src/benchmarking/results
D=$PF/data-fin
Q=$RUN/fin-queue
export ZINNIA_HARNESS_MEMORY_CAP=watchdog
DS=benchmarking/datasets/case_sets/ds1000_perf.json
NPB=benchmarking/datasets/case_sets/npbench_pythran_perf.json

ncpus() {  # number of CPUs in a taskset list such as 0-7,32-39
  local n=0 part
  for part in ${1//,/ }; do
    if [[ $part == *-* ]]; then n=$((n + ${part#*-} - ${part%-*} + 1)); else n=$((n + 1)); fi
  done
  echo $n
}
run() {  # run <label> <cpus> <case set> <shard> <systems> [extra...]
  local label=$1 cpus=$2 cs=$3 shard=$4 systems=$5; shift 5
  (cd $POST && taskset -c $cpus .venv/bin/python -m benchmarking.harness run --label $label --case-set $cs \
    --shard $shard --systems $systems --repeats 5 --threads 8 --memory-max 30G --results-dir $RES "$@" \
    > $RUN/logs/$label.log 2>&1)
  echo "$label exit $? $(date -Is) cpus $cpus"
}
# Jobs: the CPU list is always the last argument.
hw() { run fin-$1-s$2 $3 $DS $2/8 zinnia_halo2,halo2_hw,zinnia_noir,noir_hw; }
zn() { run fin-zn-s$1 $2 $NPB $1/8 zinnia_halo2,zinnia_noir --timeout 14400; }
abl() {  # abl <config> <cpus>: 3 jobs per 4 CPUs, as before (6 on 8)
  local jobs=$(( $(ncpus $2) * 3 / 4 ))
  taskset -c $2 bash $POST/benchmarking/ablations/run_configs.sh $POST $RUN/ablation-fin $1 $jobs \
    > $RUN/ablation-fin/$1.out 2>&1; echo "abl $1 exit $? $(date -Is) cpus $2"
}
fa() {  # fa <config> <jobs per 8 CPUs> <cpus>
  local jobs=$(( $(ncpus $3) * $2 / 8 ))
  taskset -c $3 bash $PF/scripts/perf_configs.sh $POST $D/perf $1 $jobs > $D/perf-$1.out 2>&1
  echo "fa $1 exit $? $(date -Is) cpus $3"
}
acc() {  # acc <name> <env or -> <cpus>; one acceptance shard per physical core only (compile limits)
  local env=$2 cpus=${3%%,*}
  [[ $env == - ]] && env=
  (cd $POST && env $env .venv/bin/python $PF/scripts/accept_sharded.py $D/accept_$1 $cpus > $RUN/logs/fin-acc-$1.log 2>&1)
  echo "acc $1 exit $? $(date -Is) cpus $cpus"
}
mb() {
  (cd $POST && taskset -c $1 .venv/bin/python benchmarking/fact_constructions/run.py --out $RUN/fact-micro/results.json \
     --jobs $(ncpus $1) --timeout 1800 > $RUN/logs/fin-mb.log 2>&1); echo "MB exit $? $(date -Is) cpus $1"
}

pop() {  # print and remove the first queued job (empty when the queue is empty)
  local line
  exec 9>$Q/lock; flock 9
  line=$(head -n 1 $Q/jobs)
  [[ -n $line ]] && sed -i 1d $Q/jobs
  flock -u 9; exec 9>&-
  printf '%s' "$line"
}
worker() {  # worker <slot> <pid to wait for, or ->
  local j=$1 phys="$((8*$1))-$((8*$1+7))" all="$((8*$1))-$((8*$1+7)),$((32+8*$1))-$((32+8*$1+7))" line kind cpus
  if [[ $2 != - ]]; then
    touch $Q/timed/slot$j
    while kill -0 $2 2>/dev/null; do sleep 20; done
    rm -f $Q/timed/slot$j
  fi
  while line=$(pop); [[ -n $line ]]; do
    set -- $line; kind=$1; shift
    if [[ $kind == timed ]]; then
      touch $Q/timed/slot$j; "$@" $phys; rm -f $Q/timed/slot$j
    else
      cpus=$phys
      [[ -z $(ls $Q/timed) ]] && cpus=$all
      "$@" $cpus
    fi
  done
}

mkdir -p $Q/timed $RUN/ablation-fin $D/perf $RUN/fact-micro
rm -f $Q/timed/*
{
  for s in 4 5 6 7; do echo "timed zn $s"; done
  for s in 0 1 2 3 4 5 6 7; do echo "timed hw h3 $s"; done
  for s in 0 1 2 3 4 5 6 7; do echo "timed hw h4 $s"; done
  # count-only jobs, longest first
  echo "untimed acc final -"
  echo "untimed acc smt ZINNIA_ABLATE_SMT=1"
  echo "untimed acc layer ZINNIA_ABLATE_LAYER_FACTS=1"
  echo "untimed acc check ZINNIA_ABLATE_CHECK_ELIM=1"
  echo "untimed acc af ZINNIA_ABLATE_ARRAY_FACTS=1"
  echo "untimed abl oracle"
  for c in final abl_joint fix_delegate fix_selects noir_hw; do echo "untimed abl $c"; done
  for c in final abl_smt abl_layer abl_check abl_af; do echo "untimed fa $c 4"; done
  echo "untimed fa noir_hw 8"
  echo "untimed mb"
} > $Q/jobs

echo "queue start $(date -Is) commit $(git -C $POST log -1 --oneline)"
for j in 0 1 2 3; do worker $j ${@:$((j+1)):1} & done
wait
echo "queue done $(date -Is)"

echo "CT start $(date -Is)"
(cd $POST && .venv/bin/python $PF/scripts/compile_timing_remote.py --repo . --out $RUN/compile-timing/ct_fin.jsonl \
   --jobs 16 --cpus 0-15 --timeout 300 --mem 8G > $RUN/logs/fin-ct.log 2>&1); echo "CT exit $? $(date -Is)"

echo "SC start $(date -Is)"
mkdir -p $D/scale/hw $D/scale-seq/hw
(cd $POST
 for f in argsort:0-7 matmul:8-15 convolve:16-23; do
   taskset -c ${f#*:} bash $PF/scripts/series_challenge.sh $POST $D/scale/challenge ${f%%:*} --time &
 done
 taskset -c 24-31 sh -c ".venv/bin/python $PF/scripts/scale_hw.py zinnia --wt $POST --data $SC --out $D/scale/hw --time" &
 wait
 for f in argsort matmul convolve; do taskset -c 0-7 bash $PF/scripts/series_challenge.sh $POST $D/scale-seq/challenge $f --time; done
 taskset -c 0-7 .venv/bin/python $PF/scripts/scale_hw.py zinnia --wt $POST --data $SC --out $D/scale-seq/hw --time
 taskset -c 0-7 .venv/bin/python $PF/scripts/scale_memory.py --out $D/scale/memory_auto_timed.jsonl --forces auto --jobs 1 --time
 taskset -c 0-63 .venv/bin/python $PF/scripts/scale_memory.py --out $D/scale/memory_cells.jsonl --jobs 8
) > $RUN/logs/fin-scale.log 2>&1; echo "SC exit $? $(date -Is)"
echo "all done $(date -Is)"
