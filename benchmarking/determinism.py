#!/usr/bin/env python3
"""Compile-determinism check: the same kernel, compiled repeatedly by the same
build in separate processes, must give byte-identical IR.

For each selected case and target (`halo2` = halo2-base, `noir`), the kernel is
compiled `--runs` times, each in a fresh Python process (so Python's string-hash
seed, Rust's `HashMap` seeds and the allocator's addresses all differ between
runs). Each run records:

* `ir`: SHA-256 of the exported IR statements (`get_ir_stmts`, canonical JSON);
* `stmts`: the number of IR statements and `ops`: a hash of the sorted
  per-operation census (equal census with a different `ir` hash means only the
  order or the numbering of statements differs);
* halo2: `cells`, the halo2-base advice + lookup cells at the chosen `k`
  (`halo2_base_cell_stats`), and `k`;
* noir: `noir`, SHA-256 of the emitted Noir source (`emit_structural` on the
  fixture's input B, as `noir_gates.py` emits it).

A case/target is **stable** when every run gives the same `ir` (and Noir source).

    .venv/bin/python benchmarking/determinism.py [--cases 0345,0400] [--runs 5]
        [--targets halo2,noir] [--jobs 6] [--out runs.json]
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "benchmarking" / "datasets" / "ds1000"
CASE_SET = ROOT / "benchmarking" / "datasets" / "case_sets" / "ds1000_perf.json"
TARGETS = ("halo2", "noir")


def perf_cases() -> list[str]:
    entries = json.loads(CASE_SET.read_text())["cases"]
    return sorted(e["case"] for e in entries if e["status"] == "perf")


def case_name(spec: str) -> str:
    return spec if spec.startswith("ds1000_case") else f"ds1000_case{int(spec):04d}"


def sha(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()[:16]


def measure(case: str, target: str) -> dict:
    """Compile once (in this process) and fingerprint the result."""
    sys.path.insert(0, str(ROOT))
    os.environ.setdefault("ZINNIA_SKIP_RELEASE_BUILD", "1")
    from benchmarking.harness.fixtures import load_fixture
    from benchmarking.harness.kernels import compile_kernel

    directory = DATASET / case
    fixture = load_fixture(directory / "fixture.json")
    compiled = compile_kernel(directory / "kernel.py", fixture.function, backend=target)
    stmts = compiled.get_ir_stmts()
    census = collections.Counter(str(s["ir_instance"]["__class__"]) for s in stmts)
    record = {
        "ir": sha(json.dumps(stmts, sort_keys=True)),
        "stmts": len(stmts),
        "ops": sha(json.dumps(sorted(census.items()))),
    }
    if target == "halo2":
        from zinnia._zinnia_core import halo2_base_cell_stats
        stats = json.loads(halo2_base_cell_stats(compiled._compiled_ir))
        record["cells"] = stats["advice"] + stats["lookup"]
        record["k"] = stats["k"]
    else:
        source = compiled.emit_structural(*fixture.arguments("B"), backend="noir")
        record["noir"] = sha(source)
        record["noir_lines"] = source.count("\n")
    return record


def run_child(case: str, target: str, timeout: float) -> dict:
    cmd = [sys.executable, str(Path(__file__).resolve()), "--child", case, target]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=ROOT)
    except subprocess.TimeoutExpired:
        return {"error": f"timeout after {timeout:.0f} s"}
    for line in reversed(proc.stdout.strip().splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
    return {"error": f"exit {proc.returncode}: {' | '.join(tail)}"[:600]}


def verdict(runs: list[dict]) -> dict:
    """Summarize the runs of one case/target: stable, or how the runs differ."""
    if any("error" in r for r in runs):
        return {"stable": False, "kind": "error", "errors": [r.get("error") for r in runs]}
    keys = ("ir", "noir") if "noir" in runs[0] else ("ir",)
    distinct = {tuple(r[k] for k in keys) for r in runs}
    if len(distinct) == 1:
        return {"stable": True}
    costs = sorted({r["cells"] for r in runs}) if "cells" in runs[0] else None
    census = {r["ops"] for r in runs}
    sizes = sorted({r["stmts"] for r in runs})
    if costs is not None and len(costs) > 1:
        kind = "cost"
    elif len(census) > 1 or len(sizes) > 1:
        kind = "census"  # different statements (Noir: gates to be measured)
    else:
        kind = "order"   # same statements, numbered or ordered differently
    return {"stable": False, "kind": kind, "variants": len(distinct), "stmts": sizes,
            "cells": costs}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--child", nargs=2, metavar=("CASE", "TARGET"), help=argparse.SUPPRESS)
    parser.add_argument("--cases", help="comma-separated case numbers (default: every perf case)")
    parser.add_argument("--targets", default=",".join(TARGETS))
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    if args.child:
        print(json.dumps(measure(*args.child)))
        return 0

    cases = [case_name(c) for c in args.cases.split(",")] if args.cases else perf_cases()
    targets = args.targets.split(",")
    jobs = [(c, t, i) for i in range(args.runs) for c in cases for t in targets]
    results: dict[str, dict[str, list[dict]]] = {c: {t: [] for t in targets} for c in cases}
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for (case, target, _), record in zip(jobs, pool.map(
                lambda job: run_child(job[0], job[1], args.timeout), jobs)):
            results[case][target].append(record)

    report = {c: {t: {"verdict": verdict(runs), "runs": runs} for t, runs in per.items()}
              for c, per in results.items()}
    unstable = [(c, t, v["verdict"]) for c, per in report.items() for t, v in per.items()
                if not v["verdict"]["stable"]]
    for case, target, v in unstable:
        print(f"UNSTABLE {case} {target}: {json.dumps(v)}")
    print(f"{len(cases) * len(targets) - len(unstable)}/{len(cases) * len(targets)} stable "
          f"over {args.runs} runs")
    if args.out:
        args.out.write_text(json.dumps(report, indent=1) + "\n")
    return 1 if unstable else 0


if __name__ == "__main__":
    sys.exit(main())
