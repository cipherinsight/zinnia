#!/usr/bin/env bash
# Task B on the final compiler, remote host shuaicpu6 (container zhantong-dev).
# Same stages as the 3ebe0558 run (scripts copied from it; only paths, worker count and the
# memory cap changed). Run in $RUN/paper-numbers-final/B after `source $RUN/env.sh`.
set -euo pipefail
cd -- "$(dirname -- "$0")"
export ZINNIA_CHECKOUT=/home/zhantong/zinnia-eval-run/zinnia-sizefix
export PYTHONDONTWRITEBYTECODE=1 B_JOBS=${B_JOBS:-16}
PY=$ZINNIA_CHECKOUT/.venv/bin/python
test "$(git -C "$ZINNIA_CHECKOUT" rev-parse HEAD)" = 2569c9edb8a2bf0b88e9351ca5752f164d2fad24
test -z "$(git -C "$ZINNIA_CHECKOUT" status --porcelain --untracked-files=no)"
test "$(sha256sum "$ZINNIA_CHECKOUT/zinnia/_zinnia_core.cpython-314-x86_64-linux-gnu.so" | cut -d' ' -f1)" \
  = 637a5c9ab0fa8f25e07745cbb0c7d9a1e395fa47ca04444de04eeecee19b1324
# No systemd in the container: every process gets a 32 GiB virtual-memory cap.
ulimit -v $((32 * 1024 * 1024))
"$PY" -B measure.py
mkdir -p probes
"$PY" -B static_audit.py
"$PY" -B run_probes.py
"$PY" -B repair_probes.py
# Final compiler only: source-audit the two probes that no longer compile (see the script).
"$PY" -B static_audit_final.py
"$PY" -B collect_arrays.py
"$PY" -B verify_inventory.py
"$PY" -B verify_original.py
"$PY" -B requirements_probe.py
"$PY" -B run_costs.py
"$PY" -B verify_costs.py
"$PY" -B report.py
