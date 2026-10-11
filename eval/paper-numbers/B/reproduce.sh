#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
export PYTHONDONTWRITEBYTECODE=1
PY=/home/zhantong/Projects/zinnia-project/wt/npb-hw/.venv/bin/python
# Each worker started by these runners is capped separately; stages are sequential.
"$PY" -B measure.py
mkdir -p probes
"$PY" -B static_audit.py
"$PY" -B run_probes.py
systemd-run --user --scope -q -p MemoryMax=16G "$PY" -B repair_probes.py
systemd-run --user --scope -q -p MemoryMax=16G "$PY" -B collect_arrays.py
"$PY" -B verify_inventory.py
"$PY" -B verify_original.py
systemd-run --user --scope -q -p MemoryMax=16G "$PY" -B requirements_probe.py
"$PY" -B run_costs.py
systemd-run --user --scope -q -p MemoryMax=16G "$PY" -B verify_costs.py
systemd-run --user --scope -q -p MemoryMax=16G "$PY" -B report.py
