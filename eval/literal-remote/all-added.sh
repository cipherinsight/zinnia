#!/usr/bin/env bash
# Timed proving + verification of the 43 programs added so that RQ2 covers every program accepted in
# RQ1 (branch all-merged). Same per-job settings as tw1 (three-way.sh): all six systems of a case in
# the same session, 3 repeats, 8 pinned cores and --threads 8 per job. All jobs start at once.
# Usage: all-added.sh <tag>
RUN=/home/zhantong/zinnia-eval-run
WT=$RUN/zinnia-literal
RES=$WT/benchmarking/results
export ZINNIA_HARNESS_MEMORY_CAP=watchdog
DS=benchmarking/datasets/case_sets/ds1000_perf.json
NPB=benchmarking/datasets/case_sets/npbench_pythran_perf.json
SYS=${SYS:-zinnia_halo2,halo2_hw,halo2_lit,zinnia_noir,noir_hw,noir_lit}
TAG=$1
run() {  # run <label> <cpus> <case set> <cases> <cap> [extra...]
  local label=$1 cpus=$2 cs=$3 cases=$4 cap=$5; shift 5
  (cd $WT && taskset -c $cpus .venv/bin/python -m benchmarking.harness run --label $label --case-set $cs \
    --cases "$cases" --systems $SYS --repeats 3 --threads 8 --memory-max $cap --results-dir $RES "$@" \
    > $RUN/logs/$label.log 2>&1)
  echo "$label exit $? $(date -Is)"
}
d() { echo "$@" | tr ' ' '\n' | sed 's/^/ds1000_case0/' | paste -sd,; }
echo "commit $(git -C $WT log -1 --oneline)  load $(cut -d' ' -f1-3 /proc/loadavg)"
run $TAG-ds-s0 0-7   $DS "$(d 291 302 303 304 313 330 337)" 26G &
run $TAG-ds-s1 8-15  $DS "$(d 339 341 342 343 349 356 357)" 26G &
run $TAG-ds-s2 16-23 $DS "$(d 365 370 373 382 386 387)" 26G &
run $TAG-ds-s3 24-31 $DS "$(d 392 393 400 401 413 417 418)" 26G &
run $TAG-ds-s4 32-39 $DS "$(d 419 430 448 449 451 456 457)" 26G &
run $TAG-ds-s5 40-47 $DS "$(d 458 463 467 468 477 502)" 26G &
run $TAG-npb-s0 48-55 $NPB npbench_deep_learning_resnet_resnet__resnet_basicblock 64G --timeout 43200 &
run $TAG-npb-s1 56-63 $NPB npbench_polybench_floyd_warshall_floyd_warshall__kernel,pythran_ccopy__CCOPY 26G --timeout 14400 &
wait
echo "all done $(date -Is)"
