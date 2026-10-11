#!/usr/bin/env bash
# Set up $RUN/zinnia-dl, the checkout of branch delegate-systems (4115cecb plus the harness
# systems zinnia_halo2_delegate / zinnia_noir_delegate; src/ identical to zinnia-post's):
# venv from the offline wheelhouse, the extension build, noir-execute and its record copied
# from zinnia-post (the container cannot fetch Noir), and check-env.
# Usage: dl-setup.sh <cpus>
RUN=/home/zhantong/zinnia-eval-run
DL=$RUN/zinnia-dl
cpus=$1
n=0; for part in ${cpus//,/ }; do [[ $part == *-* ]] && n=$((n + ${part#*-} - ${part%-*} + 1)) || n=$((n + 1)); done
set -e
cd $DL
{
  uv venv -q --python ~/.local/share/uv/python/cpython-3.14.7-linux-x86_64-gnu/bin/python3.14 .venv
  VIRTUAL_ENV=$DL/.venv uv pip install -q --offline --find-links $RUN/wheelhouse maturin==1.14.1 numpy==2.2.2 \
    pytest==8.3.4 PyYAML==6.0.3 z3-solver==4.16.0.0 setuptools wheel iniconfig packaging pluggy
  taskset -c $cpus .venv/bin/python -m benchmarking.harness build --jobs $n
  mkdir -p target/noir-execute/bin
  cp -a $RUN/zinnia-post/target/noir-execute/bin/noir-execute target/noir-execute/bin/
  cp -a $RUN/zinnia-post/target/zinnia-harness-noir-execute.json target/
  .venv/bin/python -m benchmarking.harness check-env \
    --systems zinnia_halo2,zinnia_halo2_delegate,zinnia_noir,zinnia_noir_delegate
} > $RUN/logs/dl-setup.log 2>&1
echo "dl-setup exit $? $(date -Is) cpus $cpus"
