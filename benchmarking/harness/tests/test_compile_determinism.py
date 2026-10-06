"""Compiling a kernel is deterministic: separate processes (each with its own
Python and Rust hash seeds and its own Z3 state) compile byte-identical IR and
Noir source (`benchmarking/determinism.py`).

The cases are the ones whose circuit depended on a choice that could differ
between runs: 0345 (on the Noir target, Z3 picked among equally cheap
constructions by its internal state, so two runs ordered the circuit
differently), and 0353, 0389, 0434 and 0503, whose circuits change when a Z3
query gives up (Z3 queries were bounded by wall-clock timeouts, so whether
they gave up depended on machine load; they are now bounded by step counts).
"""

from __future__ import annotations

import json
import subprocess
import sys

from benchmarking.harness import paths

SCRIPT = paths.BENCHMARKING / "determinism.py"
CASES = ("0345", "0353", "0389", "0434", "0503")


def test_separate_processes_compile_identical_circuits(tmp_path):
    out = tmp_path / "runs.json"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--cases", ",".join(CASES), "--runs", "3", "--jobs", "4",
         "--out", str(out)],
        capture_output=True, text=True, cwd=paths.REPO_ROOT, timeout=1800)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads(out.read_text())
    assert len(report) == len(CASES)
    for case, targets in report.items():
        for target, entry in targets.items():
            assert entry["verdict"] == {"stable": True}, (case, target, entry)
            assert len(entry["runs"]) == 3
