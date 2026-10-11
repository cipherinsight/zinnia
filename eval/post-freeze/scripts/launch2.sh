#!/usr/bin/env bash
# launch2.sh: launch.sh with output in $PF/data-bcc (the bcc20473 re-run; $PF/data keeps 35c86a81).
# Launch the post-freeze jobs on the remote (run on this machine). Every job is a
# `remote.sh bg` job pinned inside CPUs 0-31, refused while a Zinnia timing epoch is
# active (guard.sh check), and watched by guard.sh, which kills it if an epoch starts.
#
#   launch2.sh untimed1   acceptance final + SMT off (8 shards each, CPUs 0-7, 8-15);
#                        perf configs final/abl_smt/abl_layer/abl_check (CPUs 16-19..28-31)
#   launch2.sh untimed2   acceptance layer facts off + check elimination off (0-7, 8-15);
#                        noir_hw gates (16-23); memory series cells, all forces (24-31)
#   launch2.sh timed1     scaling series (8 threads, 30G, points in sequence, 4 series at once):
#                        argsort (0-7), matmul (8-15), convolve (16-23), scaled baselines (24-31)
#   launch2.sh timed2     memory series proofs of the compiler's choice (0-7)
#   launch2.sh timed3     repeat of timed1 with one series at a time (all on 0-7; nothing else of
#                        ours running), into $D/scale-seq
set -euo pipefail
T=/home/zhantong/Projects/zinnia-project/zinnia-src/eval/tools/remote.sh
RUN=/home/zhantong/zinnia-eval-run
PF=$RUN/post-freeze
WT=$RUN/zinnia-post
SC=$RUN/zinnia-scale
D=$PF/data-bcc
G="bash $PF/scripts/guard.sh check &&"

$T run "bash $PF/scripts/guard.sh check" || { echo "refused: a Zinnia timing epoch is active"; exit 1; }
jobs=()
bg() { $T bg "$1" "$2"; jobs+=("$1"); }
acc() {  # name env cpus
  bg "pf-acc-$1" "$G cd $WT && mkdir -p $D && env $2 .venv/bin/python $PF/scripts/accept_sharded.py $D/accept_$1 $3"
}
case $1 in
  untimed1)
    acc final "" 0-7
    acc smt "ZINNIA_ABLATE_SMT=1" 8-15
    cpus=(16-19 20-23 24-27 28-31); i=0
    for c in final abl_smt abl_layer abl_check; do
      bg "pf-perf-$c" "$G taskset -c ${cpus[$i]} bash $PF/scripts/perf_configs.sh $WT $D/perf $c 4"; i=$((i+1))
    done ;;
  untimed2)
    acc layer "ZINNIA_ABLATE_LAYER_FACTS=1" 0-7
    acc check "ZINNIA_ABLATE_CHECK_ELIM=1" 8-15
    bg pf-perf-noir_hw "$G taskset -c 16-23 bash $PF/scripts/perf_configs.sh $WT $D/perf noir_hw 8"
    bg pf-mem-cells "$G mkdir -p $D/scale && cd $WT && taskset -c 24-31 .venv/bin/python $PF/scripts/scale_memory.py --out $D/scale/memory_cells.jsonl --jobs 4" ;;
  timed1)
    for f in argsort:0-7 matmul:8-15 convolve:16-23; do
      bg "pf-scale-${f%%:*}" "$G mkdir -p $D/scale && taskset -c ${f#*:} bash $PF/scripts/series_challenge.sh $WT $D/scale/challenge ${f%%:*} --time"
    done
    bg pf-scale-hw "$G mkdir -p $D/scale/hw && cd $WT && taskset -c 24-31 sh -c '.venv/bin/python $PF/scripts/scale_hw.py zinnia --wt $WT --data $SC --out $D/scale/hw --time && .venv/bin/python $PF/scripts/scale_hw.py hw --wt $WT --data $SC --out $D/scale/hw --time'" ;;
  timed2)
    bg pf-mem-time "$G mkdir -p $D/scale && cd $WT && taskset -c 0-7 .venv/bin/python $PF/scripts/scale_memory.py --out $D/scale/memory_auto_timed.jsonl --forces auto --jobs 1 --time" ;;
  timed3)
    S=$D/scale-seq
    bg pf-scale-seq "$G mkdir -p $S/hw && cd $WT && for f in argsort matmul convolve; do taskset -c 0-7 bash $PF/scripts/series_challenge.sh $WT $S/challenge \$f --time; done && taskset -c 0-7 .venv/bin/python $PF/scripts/scale_hw.py zinnia --wt $WT --data $SC --out $S/hw --time && taskset -c 0-7 .venv/bin/python $PF/scripts/scale_hw.py hw --wt $WT --data $SC --out $S/hw --time" ;;
  *) echo "unknown stage $1" >&2; exit 2 ;;
esac
$T bg "pf-guard-$1" "bash $PF/scripts/guard.sh watch ${jobs[*]}"
