#!/usr/bin/env bash
# Create a worktree of zinnia-src on a new branch (from eval-refactor by default),
# with its own venv and a release build of the extension (both benchmark backends).
# Usage: eval-redo/tools/new-worktree.sh <name> [base-branch]
set -euo pipefail
name="$1"; base="${2:-eval-refactor}"
root=/home/zhantong/Projects/zinnia-project
wt="$root/wt/$name"
git -C "$root/zinnia-src" worktree add -q -b "$name" "$wt" "$base"
cd "$wt"
uv venv -q --python /usr/bin/python3.14 .venv
VIRTUAL_ENV="$wt/.venv" uv pip install -q maturin==1.14.1 numpy==2.2.2 pytest==8.3.4 \
  PyYAML==6.0.3 z3-solver==4.16.0.0 setuptools wheel iniconfig packaging pluggy
RUSTC_BOOTSTRAP=1 VIRTUAL_ENV="$wt/.venv" PYTHON=.venv/bin/python CARGO_BUILD_JOBS="${JOBS:-6}" \
  nice -n 10 scripts/develop.sh --release --features halo2-base-backend,noir-backend \
  > "$root/wt/$name.build.log" 2>&1 || { tail -20 "$root/wt/$name.build.log"; exit 1; }
.venv/bin/python -c 'import zinnia._zinnia_core as c; print("built", c.__file__)'
