"""Compiling a kernel is deterministic: separate processes (each with its own
Python and Rust hash seeds and its own Z3 state) compile byte-identical IR and
Noir source (`benchmarking/determinism.py`).

DS-1000 cases whose circuit depended on a choice that could differ between runs:
0345 (on the Noir target, Z3 picked among equally cheap constructions by its
internal state, so two runs ordered the circuit differently), and 0353, 0389,
0434 and 0503, whose circuits change when a Z3 query gives up (Z3 queries were
bounded by wall-clock timeouts, so whether they gave up depended on machine
load; they are now bounded by step counts).

NPBench/Pythran entries with data-dependent branches: Pythran `crotg` merged
the variables its two branches assign in hash-map order, so each run ordered
the merging selects differently; the others branch on runtime values too.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from benchmarking.harness import paths

SCRIPT = paths.BENCHMARKING / "determinism.py"
SAMPLES = {
    "ds1000_perf.json": ("0345", "0353", "0389", "0434", "0503"),
    "npbench_pythran_perf.json": ("pythran_crotg__CROTG", "pythran_clip__clip",
                                  "pythran_bubble_sort__bubble_sort", "pythran_ising__ising_step",
                                  "pythran_perm__permutations"),
}


@pytest.mark.parametrize("case_set", sorted(SAMPLES))
def test_separate_processes_compile_identical_circuits(tmp_path, case_set):
    out = tmp_path / "runs.json"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--case-set", str(paths.CASE_SETS / case_set),
         "--cases", ",".join(SAMPLES[case_set]), "--runs", "3", "--jobs", "4", "--out", str(out)],
        capture_output=True, text=True, cwd=paths.REPO_ROOT, timeout=1800)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads(out.read_text())
    assert len(report) == len(SAMPLES[case_set])
    for case, targets in report.items():
        for target, entry in targets.items():
            assert entry["verdict"]["stable"], (case, target, entry)
            assert len(entry["runs"]) == 3
