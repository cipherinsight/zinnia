#!/usr/bin/env bash
# Task A on the final compiler, remote host shuaicpu6 (container zhantong-dev).
# Same pipeline as the 3ebe0558 run (scripts copied from it; only paths, worker count and the
# memory cap changed): stock compile + phase0 (auxiliary), instrumented compiler copy, stock vs
# instrumented audit, mask-compaction log, classification, report.
# Run in $RUN/paper-numbers-final/A after `source $RUN/env.sh`.
set -euo pipefail
cd "$(dirname "$0")"
export ZINNIA_CHECKOUT=/home/zhantong/zinnia-eval-run/zinnia-sizefix
ROOT=$ZINNIA_CHECKOUT
PY="$ROOT/.venv/bin/python"
export A_JOBS=${A_JOBS:-16}
export PYTHONDONTWRITEBYTECODE=1
export PYO3_PYTHON="$PY"
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
export CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_DEV_OPT_LEVEL=1
Z3_DIR="$($PY -c 'import z3,pathlib; print(pathlib.Path(z3.__file__).parent / "lib")')"
export LIBRARY_PATH="$Z3_DIR${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$Z3_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
test "$(git -C "$ROOT" rev-parse HEAD)" = 2569c9edb8a2bf0b88e9351ca5752f164d2fad24
test -z "$(git -C "$ROOT" status --porcelain --untracked-files=no)"
test "$(sha256sum "$ROOT/zinnia/_zinnia_core.cpython-314-x86_64-linux-gnu.so" | cut -d' ' -f1)" \
  = 637a5c9ab0fa8f25e07745cbb0c7d9a1e395fa47ca04444de04eeecee19b1324
"$PY" instrument.py
cargo build --manifest-path compiler/Cargo.toml --no-default-features --features extension-module -j 16 > build.log 2>&1
cp compiler/target/debug/lib_zinnia_core.so compiler/zinnia/_zinnia_core.so
# No systemd in the container: every compilation process gets a 32 GiB virtual-memory cap.
ulimit -v $((32 * 1024 * 1024))
"$PY" analyze.py > run.log 2>&1
"$PY" audit.py stock > stock.log 2>&1
"$PY" audit.py instrumented > instrumented.log 2>&1
"$PY" compaction.py > compaction.log 2>&1
"$PY" summarize.py > summary.txt
"$PY" report.py
