#!/usr/bin/env python3
"""Compile-determinism check: the same kernel, compiled repeatedly by the same
build in separate processes, must give byte-identical IR.

For each selected case of a performance case set (default the DS-1000 set;
`--case-set` takes another, e.g. `npbench_pythran_perf.json`, whose entries name
their folder and entry function) and target (`halo2` = halo2-base, `noir`), the kernel is
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

    .venv/bin/python benchmarking/determinism.py [--case-set PATH] [--cases 0345,0400]
        [--runs 5] [--targets halo2,noir] [--jobs 6] [--out runs.json]
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
BENCHMARKING = ROOT / "benchmarking"
DS1000_CASE_SET = BENCHMARKING / "datasets" / "case_sets" / "ds1000_perf.json"
TARGETS = ("halo2", "noir")


def perf_entries(case_set: Path) -> dict[str, dict]:
    """The case set's `perf` entries by case name."""
    entries = json.loads(case_set.read_text())["cases"]
    return {e["case"]: e for e in entries if e["status"] == "perf"}


def case_name(spec: str, entries: dict[str, dict]) -> str:
    """A case name, or a DS-1000 case number (`345` or `0345`)."""
    if spec in entries or not spec.isdigit():
        return spec
    return f"ds1000_case{int(spec):04d}"


def case_files(entry: dict) -> tuple[Path, Path, str]:
    """`(kernel.py, fixture.json, entry function)` of a case-set entry: its
    `path` folder (relative to `benchmarking/`) or `datasets/ds1000/<case>`."""
    directory = (BENCHMARKING / entry["path"] if "path" in entry
                 else BENCHMARKING / "datasets" / "ds1000" / entry["case"])
    fixture = directory / (entry.get("fixture") or "fixture.json")
    return directory / "kernel.py", fixture, entry.get("function", "kernel")


def sha(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()[:16]


def measure(kernel: str, fixture_path: str, function: str, target: str) -> dict:
    """Compile once (in this process) and fingerprint the result."""
    sys.path.insert(0, str(ROOT))
    os.environ.setdefault("ZINNIA_SKIP_RELEASE_BUILD", "1")
    from benchmarking.harness.fixtures import load_fixture
    from benchmarking.harness.kernels import compile_kernel

    fixture = load_fixture(Path(fixture_path))
    compiled = compile_kernel(Path(kernel), function, backend=target)
    stmts = compiled.get_ir_stmts()
    census = collections.Counter(str(s["ir_instance"]["__class__"]) for s in stmts)
    record = {
        "ir": sha(json.dumps(stmts, sort_keys=True)),
        "stmts": len(stmts),
        "ops": sha(json.dumps(sorted(census.items()))),
    }
    try:
        if target == "halo2":
            from zinnia._zinnia_core import halo2_base_cell_stats
            stats = json.loads(halo2_base_cell_stats(compiled._compiled_ir))
            record["cells"] = stats["advice"] + stats["lookup"]
            record["k"] = stats["k"]
        else:
            source = compiled.emit_structural(*fixture.arguments("B"), backend="noir")
            record["noir"] = sha(source)
            record["noir_lines"] = source.count("\n")
    except Exception as exc:  # noqa: BLE001 - a backend that rejects the IR (the IR still counts)
        record["backend_error"] = f"{type(exc).__name__}: {exc}"[:300]
    return record


def run_child(entry: dict, target: str, timeout: float) -> dict:
    kernel, fixture, function = case_files(entry)
    cmd = [sys.executable, str(Path(__file__).resolve()), "--child", str(kernel), str(fixture),
           function, target]
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
    """Summarize the runs of one case/target: stable, or how the runs differ.
    A compilation that fails the same way on every run is deterministic: it is
    stable, with the error. A backend that rejects the IR (`backend_error`)
    still leaves the IR to compare."""
    errors = {r.get("error") for r in runs}
    if errors != {None}:
        if len(errors) == 1:
            return {"stable": True, "error": runs[0]["error"]}
        return {"stable": False, "kind": "error", "errors": [r.get("error") for r in runs]}
    keys = ("ir", "noir", "backend_error")
    distinct = {tuple(r.get(k) for k in keys) for r in runs}
    if len(distinct) == 1:
        return {"stable": True, **({"backend_error": runs[0]["backend_error"]}
                                   if "backend_error" in runs[0] else {})}
    costs = sorted({r["cells"] for r in runs if "cells" in r}) or None
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
    parser.add_argument("--child", nargs=4, metavar=("KERNEL", "FIXTURE", "FUNCTION", "TARGET"),
                        help=argparse.SUPPRESS)
    parser.add_argument("--case-set", type=Path, default=DS1000_CASE_SET,
                        help="performance case set (default: the DS-1000 set)")
    parser.add_argument("--cases", help="comma-separated case names or DS-1000 numbers "
                        "(default: every perf case of the case set)")
    parser.add_argument("--targets", default=",".join(TARGETS))
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    if args.child:
        print(json.dumps(measure(*args.child)))
        return 0

    entries = perf_entries(args.case_set)
    cases = ([case_name(c, entries) for c in args.cases.split(",")] if args.cases
             else sorted(entries))
    unknown = [c for c in cases if c not in entries]
    if unknown:
        parser.error(f"not perf cases of {args.case_set.name}: {', '.join(unknown)}")
    targets = args.targets.split(",")
    jobs = [(c, t, i) for i in range(args.runs) for c in cases for t in targets]
    results: dict[str, dict[str, list[dict]]] = {c: {t: [] for t in targets} for c in cases}
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for (case, target, _), record in zip(jobs, pool.map(
                lambda job: run_child(entries[job[0]], job[1], args.timeout), jobs)):
            results[case][target].append(record)

    report = {c: {t: {"verdict": verdict(runs), "runs": runs} for t, runs in per.items()}
              for c, per in results.items()}
    unstable = [(c, t, v["verdict"]) for c, per in report.items() for t, v in per.items()
                if not v["verdict"]["stable"]]
    for case, target, v in unstable:
        print(f"UNSTABLE {case} {target}: {json.dumps(v)}")
    failing = sum(1 for per in report.values() for v in per.values()
                  if {"error", "backend_error"} & v["verdict"].keys())
    print(f"{len(cases) * len(targets) - len(unstable)}/{len(cases) * len(targets)} stable "
          f"over {args.runs} runs ({failing} of them fail the same way on every run)")
    if args.out:
        args.out.write_text(json.dumps(report, indent=1) + "\n")
    return 1 if unstable else 0


if __name__ == "__main__":
    sys.exit(main())
