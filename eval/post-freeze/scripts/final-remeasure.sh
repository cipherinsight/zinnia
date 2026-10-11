#!/usr/bin/env bash
# Full re-measurement of the final compiler (array facts + fact-driven lowering rules) on CPUs 0-31,
# from $RUN/zinnia-post. zkVM epochs are not re-run (their programs did not change): Zinnia's
# DS-1000 and NPBench/Pythran runs are paired with the ff1 zkVM epochs.
#   H1..H4  paired hand-written sessions (zinnia_halo2, halo2_hw, zinnia_noir, noir_hw; x5; Noir
#           witness by noir-execute)            -> results/<date>-fin-hN-s*
#   ZD      Zinnia on DS-1000 perf x5 (halo2, Noir)            -> fin-zd-s*
#   ZN      Zinnia on NPBench/Pythran perf x5 (halo2, Noir)    -> fin-zn-s*
#   CS      circuit size + array-lowering ablations + oracle + noir_hw -> $RUN/ablation-fin
#   FA      fact ablations (SMT, layer facts, check elim, array facts) -> $PF/data-fin/perf
#   AC      acceptance: final and the four fact ablations     -> $PF/data-fin/accept_*
#   CT      compile time                                      -> $RUN/compile-timing/ct_fin.jsonl
#   MB      array-fact micro-benchmark sweeps                  -> $RUN/fact-micro/results.json
#   SC      scaling (parallel run, sequential run, memory series)   -> $PF/data-fin/scale*
# Usage: final-remeasure.sh [stages...]   (default: H1 ZD H2 ZN H3 CS FA AC CT MB SC H4)
RUN=/home/zhantong/zinnia-eval-run
POST=$RUN/zinnia-post
PF=$RUN/post-freeze
SC=$RUN/zinnia-scale
RES=$RUN/zinnia-src/benchmarking/results
D=$PF/data-fin
export ZINNIA_HARNESS_MEMORY_CAP=watchdog
DS=benchmarking/datasets/case_sets/ds1000_perf.json
NPB=benchmarking/datasets/case_sets/npbench_pythran_perf.json
STAGES=${*:-H1 ZD H2 ZN H3 CS FA AC CT MB SC H4}
run() {  # run <label> <cpus> <case set> <shard> <systems> [extra...]
  local label=$1 cpus=$2 cs=$3 shard=$4 systems=$5; shift 5
  (cd $POST && taskset -c $cpus .venv/bin/python -m benchmarking.harness run --label $label --case-set $cs \
    --shard $shard --systems $systems --repeats 5 --threads 8 --memory-max 30G --results-dir $RES "$@" \
    > $RUN/logs/$label.log 2>&1)
  echo "$label exit $? $(date -Is)"
}
waves() {  # waves <function> <tag>: shards 0-3 then 4-7, 8 pinned cores each
  for first in 0 4; do
    pids=()
    for j in 0 1 2 3; do $1 $((first + j)) $((8*j))-$((8*j+7)) $2 & pids+=($!); done
    for p in "${pids[@]}"; do wait $p; done
  done
}
hw() { run fin-$3-s$1 $2 $DS $1/8 zinnia_halo2,halo2_hw,zinnia_noir,noir_hw; }
zd() { run fin-zd-s$1 $2 $DS $1/8 zinnia_halo2,zinnia_noir; }
zn() { run fin-zn-s$1 $2 $NPB $1/8 zinnia_halo2,zinnia_noir --timeout 14400; }
abl() { taskset -c $2 bash $POST/benchmarking/ablations/run_configs.sh $POST $RUN/ablation-fin $1 $3 \
          > $RUN/ablation-fin/$1.out 2>&1; echo "abl $1 exit $? $(date -Is)"; }
fa() { taskset -c $2 bash $PF/scripts/perf_configs.sh $POST $D/perf $1 $3 > $D/perf-$1.out 2>&1; echo "fa $1 exit $? $(date -Is)"; }
acc() { (cd $POST && env $2 .venv/bin/python $PF/scripts/accept_sharded.py $D/accept_$1 $3 > $RUN/logs/fin-acc-$1.log 2>&1)
        echo "acc $1 exit $? $(date -Is)"; }
echo "commit $(git -C $POST log -1 --oneline)"
for s in $STAGES; do
  echo "$s start $(date -Is)"
  case $s in
    H*) waves hw ${s,,} ;;
    ZD) waves zd x ;;
    ZN) waves zn x ;;
    CS) mkdir -p $RUN/ablation-fin
        abl final 0-7 6 & abl abl_joint 8-15 6 & abl fix_delegate 16-23 6 & abl fix_selects 24-31 6 & wait
        abl oracle 0-23 18 & abl noir_hw 24-31 8 & wait ;;
    FA) mkdir -p $D/perf
        fa final 0-7 4 & fa abl_smt 8-15 4 & fa abl_layer 16-23 4 & fa abl_check 24-31 4 & wait
        fa abl_af 0-7 4 & fa noir_hw 8-15 8 & wait ;;
    AC) mkdir -p $D
        acc final "" 0-7 & acc smt "ZINNIA_ABLATE_SMT=1" 8-15 & acc layer "ZINNIA_ABLATE_LAYER_FACTS=1" 16-23 & acc check "ZINNIA_ABLATE_CHECK_ELIM=1" 24-31 & wait
        acc af "ZINNIA_ABLATE_ARRAY_FACTS=1" 0-15 ;;
    CT) (cd $POST && .venv/bin/python $PF/scripts/compile_timing_remote.py --repo . --out $RUN/compile-timing/ct_fin.jsonl \
           --jobs 16 --cpus 0-15 --timeout 300 --mem 8G > $RUN/logs/fin-ct.log 2>&1); echo "CT exit $? $(date -Is)" ;;
    MB) mkdir -p $RUN/fact-micro
        (cd $POST && taskset -c 0-31 .venv/bin/python benchmarking/fact_constructions/run.py --out $RUN/fact-micro/results.json \
           --jobs 16 --timeout 1800 > $RUN/logs/fin-mb.log 2>&1); echo "MB exit $? $(date -Is)" ;;
    SC) mkdir -p $D/scale/hw $D/scale-seq/hw
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
        ) > $RUN/logs/fin-scale.log 2>&1; echo "SC exit $? $(date -Is)" ;;
  esac
done
echo "all done $(date -Is)"
