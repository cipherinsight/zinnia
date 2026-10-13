#!/usr/bin/env bash
# Tightness of the capacities of Task B (see REPORT.md). Local, hardware-independent:
# no keys, no proofs (mock constraint checks only), no timings.
# ZINNIA_CHECKOUT: a zinnia-src checkout with a built extension (defaults to this repository).
set -euo pipefail
cd -- "$(dirname -- "$0")"
PY=${PY:-$(cd ../../../.. && pwd)/.venv/bin/python}
export PYTHONDONTWRITEBYTECODE=1
"$PY" -B run_search.py      # methods (a) and (b): 8 processes, each under a 16 GB cap
systemd-run --user --scope -q -p MemoryMax=16G "$PY" -B smt.py > smt.log   # method (c)
"$PY" -B validate_mock.py   # worst-case inputs on the compiled circuits (mock backend)
"$PY" -B report.py          # tightness.csv, summary.json, generated tables of REPORT.md
