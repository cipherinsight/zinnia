#!/usr/bin/env bash
# After freeze-timing: everything on CPUs 0-31, from $RUN/zinnia-post (eval-refactor bcc20473: the
# bc584e01 circuits on every perf case, plus the noir-execute harness and the ablation switches).
#   NX  three Noir-only paired sessions (zinnia_noir, noir_hw; repeats 5), witness by noir-execute
#   AB  ablations, compile only: final, abl_joint, fix_delegate, fix_selects (4 x 8 CPUs), then
#       oracle (0-23) and noir_hw (24-31); output in $RUN/ablation-bcc
# Usage: post-timing.sh [stages...]   (default: NX AB)
RUN=/home/zhantong/zinnia-eval-run
POST=$RUN/zinnia-post
RES=$RUN/zinnia-src/benchmarking/results
OUT=$RUN/ablation-bcc
export ZINNIA_HARNESS_MEMORY_CAP=watchdog
DS=benchmarking/datasets/case_sets/ds1000_perf.json
STAGES=${*:-NX AB}
noir_shard() {  # noir_shard <shard> <cpus> <tag>
  local label=$3-s$1
  (cd $POST && taskset -c $2 .venv/bin/python -m benchmarking.harness run --label $label --case-set $DS \
    --shard $1/8 --systems zinnia_noir,noir_hw --repeats 5 --threads 8 --memory-max 30G --results-dir $RES \
    > $RUN/logs/$label.log 2>&1)
  echo "$label exit $? $(date -Is)"
}
waves() {  # waves <tag>: shards 0-3 then 4-7, 8 pinned cores each within 0-31
  for first in 0 4; do
    pids=()
    for j in 0 1 2 3; do noir_shard $((first + j)) $((8*j))-$((8*j+7)) $1 & pids+=($!); done
    for p in "${pids[@]}"; do wait $p; done
  done
}
abl() {  # abl <config> <cpus> <jobs>
  taskset -c $2 bash $POST/benchmarking/ablations/run_configs.sh $POST $OUT $1 $3 > $OUT/$1.out 2>&1
  echo "abl $1 exit $? $(date -Is)"
}
echo "commit $(git -C $POST log -1 --oneline)"
for s in $STAGES; do
  echo "$s start $(date -Is)"
  case $s in
    NX) for t in nx1 nx2 nx3; do waves $t; done ;;
    AB) mkdir -p $OUT
        abl final 0-7 6 & abl abl_joint 8-15 6 & abl fix_delegate 16-23 6 & abl fix_selects 24-31 6 & wait
        abl oracle 0-23 18 & abl noir_hw 24-31 8 & wait ;;
  esac
done
echo "all done $(date -Is)"
