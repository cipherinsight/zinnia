#!/usr/bin/env bash
# Re-runs after the cost-model fix (eval-refactor d9b0351c in $RUN/zinnia-post), all on CPUs 0-31.
# Only 3 halo2-base perf circuits differ from bc584e01: ds1000_case0375, npbench softmax,
# pythran make_decision (Noir identical).
#   RT   re-time them, nothing else of ours running: 0375 in three paired sessions (rt1-rt3: the four
#        hand-written-comparison systems, repeats 5); the two NPBench/Pythran cases with Zinnia
#        halo2-base x5 (rtn; the zkVM times of ff1-npb-zkvm stay valid)
#   SC   scaling series again (the challenge price changed): parallel run then the sequential run
#   AB   array-lowering ablations + oracle + noir_hw into $RUN/ablation-cm
#   FA   fact-layer ablations + acceptance (final) into $PF/data-cm
# Usage: cm-rerun.sh [stages...]   (default: RT SC AB FA)
RUN=/home/zhantong/zinnia-eval-run
POST=$RUN/zinnia-post
PF=$RUN/post-freeze
SC=$RUN/zinnia-scale
RES=$RUN/zinnia-src/benchmarking/results
export ZINNIA_HARNESS_MEMORY_CAP=watchdog
NPB=benchmarking/datasets/case_sets/npbench_pythran_perf.json
STAGES=${*:-RT SC AB FA}
harness() {  # harness <label> <cpus> <args...>
  local label=$1 cpus=$2; shift 2
  (cd $POST && taskset -c $cpus .venv/bin/python -m benchmarking.harness run --label $label --threads 8 \
    --memory-max 30G --results-dir $RES "$@" > $RUN/logs/$label.log 2>&1)
  echo "$label exit $? $(date -Is)"
}
abl() {  # abl <config> <cpus> <jobs>
  taskset -c $2 bash $POST/benchmarking/ablations/run_configs.sh $POST $RUN/ablation-cm $1 $3 \
    > $RUN/ablation-cm/$1.out 2>&1
  echo "abl $1 exit $? $(date -Is)"
}
fa() {  # fa <config> <cpus> <jobs>
  taskset -c $2 bash $PF/scripts/perf_configs.sh $POST $PF/data-cm/perf $1 $3 > $PF/data-cm/perf-$1.out 2>&1
  echo "fa $1 exit $? $(date -Is)"
}
echo "commit $(git -C $POST log -1 --oneline)"
for s in $STAGES; do
  echo "$s start $(date -Is)"
  case $s in
    RT) for t in rt1 rt2 rt3; do
          harness $t 0-7 --cases 0375 --systems zinnia_halo2,halo2_hw,zinnia_noir,noir_hw --repeats 5
        done
        harness rtn 0-7 --case-set $NPB --cases npbench_deep_learning_softmax_softmax__softmax,pythran_make_decision__md \
          --systems zinnia_halo2 --repeats 5 --timeout 14400 ;;
    SC) D=$PF/data-cm; mkdir -p $D/scale/hw $D/scale-seq/hw
        (cd $POST
         for f in argsort:0-7 matmul:8-15 convolve:16-23; do
           taskset -c ${f#*:} bash $PF/scripts/series_challenge.sh $POST $D/scale/challenge ${f%%:*} --time &
         done
         taskset -c 24-31 sh -c ".venv/bin/python $PF/scripts/scale_hw.py zinnia --wt $POST --data $SC --out $D/scale/hw --time" &
         wait
         for f in argsort matmul convolve; do taskset -c 0-7 bash $PF/scripts/series_challenge.sh $POST $D/scale-seq/challenge $f --time; done
         taskset -c 0-7 .venv/bin/python $PF/scripts/scale_hw.py zinnia --wt $POST --data $SC --out $D/scale-seq/hw --time
         taskset -c 0-7 .venv/bin/python $PF/scripts/scale_memory.py --out $D/scale/memory_auto_timed.jsonl --forces auto --jobs 1 --time
         taskset -c 0-31 .venv/bin/python $PF/scripts/scale_memory.py --out $D/scale/memory_cells.jsonl --jobs 4
        ) > $RUN/logs/cm-scale.log 2>&1
        echo "SC exit $? $(date -Is)" ;;
    AB) mkdir -p $RUN/ablation-cm
        abl final 0-7 6 & abl abl_joint 8-15 6 & abl fix_delegate 16-23 6 & abl fix_selects 24-31 6 & wait
        abl oracle 0-23 18 & abl noir_hw 24-31 8 & wait ;;
    FA) mkdir -p $PF/data-cm/perf
        fa final 0-7 4 & fa abl_smt 8-15 4 & fa abl_layer 16-23 4 & fa abl_check 24-31 4 & wait
        fa noir_hw 0-7 8 &
        (cd $POST && env .venv/bin/python $PF/scripts/accept_sharded.py $PF/data-cm/accept_final 8-23 \
           > $RUN/logs/cm-accept.log 2>&1; echo "accept exit $? $(date -Is)") &
        wait ;;
  esac
done
echo "all done $(date -Is)"
