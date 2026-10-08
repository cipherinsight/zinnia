#!/usr/bin/env bash
# Set up the scaling study's checkout $RUN/zinnia-scnpb (branch scaling-npb: 4115cecb's src/ plus
# the delegate harness systems and benchmarking/scaling_npb), as dl-setup.sh set up zinnia-dl:
# venv from the offline wheelhouse, the extension build, noir-execute and its record copied from
# zinnia-post (the container cannot fetch Noir), the SP1 and RISC Zero hosts and guests (target
# directories seeded from $ZK_SEED), the data directory's case sets pointed at $DATA, the zkVM
# programs diffed against the originals, check-env. No timing.
# Usage: setup.sh <cpus>     (CPUs 32-63 only while no timed job runs anywhere)
source "$(dirname "$0")/env.sh"
ZK_SEED=${ZK_SEED:-$RUN/zinnia-src}
cpus=$1 n=$(ncpus $1)
set -e
cd $CO
mkdir -p $OUT/logs $OUT/counts/parts $OUT/validate $OUT/results
{
  git diff --quiet 4115cecb -- src || { echo "src/ differs from 4115cecb"; exit 1; }
  uv venv -q --python ~/.local/share/uv/python/cpython-3.14.7-linux-x86_64-gnu/bin/python3.14 .venv
  VIRTUAL_ENV=$CO/.venv uv pip install -q --offline --find-links $RUN/wheelhouse maturin==1.14.1 numpy==2.2.2 \
    pytest==8.3.4 PyYAML==6.0.3 z3-solver==4.16.0.0 setuptools wheel iniconfig packaging pluggy
  taskset -c $cpus $PY -m benchmarking.harness build --jobs $n
  mkdir -p target/noir-execute/bin
  cp -a $RUN/zinnia-post/target/noir-execute/bin/noir-execute target/noir-execute/bin/
  cp -a $RUN/zinnia-post/target/zinnia-harness-noir-execute.json target/
  for s in sp1 risc0; do
    t=benchmarking/baselines/$s/target
    [[ -d $t || ! -d $ZK_SEED/$t ]] || cp -r --reflink=auto $ZK_SEED/$t $t
  done
  taskset -c $cpus $PY -m benchmarking.harness build --skip-zinnia --baselines sp1,risc0 --jobs $n
  $PY -m benchmarking.scaling_npb relocate --data $DATA
  $PY -m benchmarking.scaling_npb zkvm-check --data $DATA --out $OUT/zkvm_check.json | tail -1
  echo "/dev/shm: $(df -h /dev/shm | tail -1)"   # SP1 died with SIGBUS at Docker's 64 MB default
  $PY -m benchmarking.harness check-env \
    --systems zinnia_halo2,zinnia_halo2_delegate,zinnia_noir,zinnia_noir_delegate,sp1,risc0 \
    --out $OUT/check-env.json
} > $OUT/logs/setup.log 2>&1
echo "scnpb setup exit $? $(date -Is) cpus $cpus"
