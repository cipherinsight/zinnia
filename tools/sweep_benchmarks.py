#!/usr/bin/env python3
"""
Benchmark sweep — try to compile every Python file under benchmarking/*/ and
report PASS / TIMEOUT / FAIL.

Usage:
    python tools/sweep_benchmarks.py [--timeout 60] [--smt {on,off,both}]
                                     [--filter <regex>]
                                     [--out <path>]
                                     [--workers <int>]

--smt off    : run with ZINNIA_SMT_ENABLE=0 (the env-var safety net wired
               into the Rust `compile_circuit` entry point).
--smt on     : run with the default (smt_enable=true).
--smt both   : run twice and emit a delta table.

The harness imports each benchmark file, finds every `@zk_circuit`-decorated
callable in the module, and calls `.compile()` on a `ZKCircuit` constructed
from its source. PASS = compile returned without exception. FAIL =
exception (with class + first message line). TIMEOUT = subprocess hit the
wall clock limit.

Each benchmark runs in its own subprocess so a hard panic in the Rust
core can't take down the whole sweep.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_GLOB = "benchmarking/*/*.py"


def _venv_python() -> str:
    """Pick the Python interpreter that has the zinnia package installed."""
    candidate = REPO_ROOT / ".venv" / "bin" / "python"
    if candidate.exists():
        return str(candidate)
    return sys.executable


PYTHON = _venv_python()


# Tiny driver that runs inside each subprocess. We pass it as a `-c` string
# to avoid creating a temp file. It prints ONE JSON object on stdout.
DRIVER = r"""
import json, sys, traceback, importlib.util, inspect, os

bench_path = sys.argv[1]
bench_dir = os.path.dirname(bench_path)
bench_stem = os.path.splitext(os.path.basename(bench_path))[0]

sys.path.insert(0, bench_dir)

t0 = __import__('time').time()
try:
    spec = importlib.util.spec_from_file_location(bench_stem, bench_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
except BaseException as e:
    print(json.dumps({
        "status": "FAIL",
        "phase": "import",
        "exc": type(e).__name__,
        "msg": str(e).splitlines()[0] if str(e) else "",
        "elapsed": __import__('time').time() - t0,
    }))
    sys.exit(0)

# Find @zk_circuit-decorated callables. They are wrappers named
# `__zk_circuit_annotator_inner` whose source lives in their closure.
from zinnia.api.zk_circuit import ZKCircuit

candidates = []
for name, obj in vars(mod).items():
    if name.startswith("_"):
        continue
    if callable(obj) and getattr(obj, "__name__", None) == "__zk_circuit_annotator_inner":
        candidates.append((name, obj))

if not candidates:
    print(json.dumps({
        "status": "FAIL",
        "phase": "discover",
        "exc": "NoCircuit",
        "msg": "no @zk_circuit decorated callable found",
        "elapsed": __import__('time').time() - t0,
    }))
    sys.exit(0)

# Compile each candidate. First one that fails wins; otherwise PASS.
for name, fn in candidates:
    t1 = __import__('time').time()
    try:
        circuit = ZKCircuit.from_method(fn)
        circuit.compile()
    except BaseException as e:
        print(json.dumps({
            "status": "FAIL",
            "phase": "compile",
            "circuit": name,
            "exc": type(e).__name__,
            "msg": str(e).splitlines()[0] if str(e) else "",
            "elapsed": __import__('time').time() - t0,
            "compile_elapsed": __import__('time').time() - t1,
        }))
        sys.exit(0)

print(json.dumps({
    "status": "PASS",
    "elapsed": __import__('time').time() - t0,
    "circuits": [n for n, _ in candidates],
}))
"""


@dataclass
class Outcome:
    name: str
    status: str  # PASS | TIMEOUT | FAIL
    detail: str = ""  # Exception class / failure-phase summary
    elapsed: float = 0.0
    raw: dict = field(default_factory=dict)

    @property
    def bucket_label(self) -> str:
        if self.status == "PASS":
            return "PASS"
        if self.status == "TIMEOUT":
            return "TIMEOUT"
        return f"FAIL:{self.detail}" if self.detail else "FAIL"


def discover_benchmarks(filter_re: str | None) -> list[Path]:
    paths = sorted(REPO_ROOT.glob(BENCHMARK_GLOB))
    out: list[Path] = []
    pat = re.compile(filter_re) if filter_re else None
    for p in paths:
        if p.name in ("__init__.py", "conftest.py"):
            continue
        if pat and not pat.search(p.stem):
            continue
        out.append(p)
    return out


def run_one(bench: Path, timeout: float, smt_enable: bool) -> Outcome:
    env = os.environ.copy()
    env["ZINNIA_SMT_ENABLE"] = "1" if smt_enable else "0"
    # Suppress pyc clutter in benchmark dirs.
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    t0 = time.time()
    try:
        proc = subprocess.run(
            [PYTHON, "-c", DRIVER, str(bench)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(REPO_ROOT),
            env=env,
        )
    except subprocess.TimeoutExpired:
        return Outcome(
            name=bench.parent.name + "/" + bench.name,
            status="TIMEOUT",
            detail="",
            elapsed=time.time() - t0,
        )
    elapsed = time.time() - t0
    name = bench.parent.name + "/" + bench.name
    raw: dict[str, Any] = {}
    # The driver always emits ONE JSON line on stdout — find it.
    line = ""
    for cand in proc.stdout.splitlines():
        cand = cand.strip()
        if cand.startswith("{") and cand.endswith("}"):
            line = cand
    if line:
        try:
            raw = json.loads(line)
        except Exception:
            raw = {}
    if proc.returncode != 0 and not raw:
        # Driver crashed without printing JSON — likely a Rust panic that
        # reached the process boundary, or a SIGABRT.
        msg = (proc.stderr or "").strip().splitlines()
        last = msg[-1] if msg else f"rc={proc.returncode}"
        return Outcome(
            name=name,
            status="FAIL",
            detail=f"DriverCrash:{last[:120]}",
            elapsed=elapsed,
            raw={"stderr": (proc.stderr or "")[-400:]},
        )
    status = raw.get("status", "FAIL")
    if status == "PASS":
        return Outcome(name=name, status="PASS", elapsed=elapsed, raw=raw)
    detail = raw.get("exc", "Unknown")
    return Outcome(name=name, status="FAIL", detail=detail, elapsed=elapsed, raw=raw)


def sweep(
    benches: list[Path],
    timeout: float,
    smt_enable: bool,
    workers: int,
) -> list[Outcome]:
    out: list[Outcome] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_one, b, timeout, smt_enable): b for b in benches}
        for i, fut in enumerate(as_completed(futs), 1):
            o = fut.result()
            out.append(o)
            sys.stderr.write(
                f"  [{i:3d}/{len(benches)}] {o.status:<7} {o.name}"
                + (f"  ({o.detail})" if o.detail else "")
                + f"  {o.elapsed:5.1f}s\n"
            )
            sys.stderr.flush()
    out.sort(key=lambda o: o.name)
    return out


def summarise(label: str, outcomes: list[Outcome]) -> dict:
    total = len(outcomes)
    by_bucket: dict[str, int] = {}
    by_exc: dict[str, int] = {}
    pass_n = timeout_n = fail_n = 0
    for o in outcomes:
        if o.status == "PASS":
            pass_n += 1
        elif o.status == "TIMEOUT":
            timeout_n += 1
        else:
            fail_n += 1
            by_exc[o.detail] = by_exc.get(o.detail, 0) + 1
        by_bucket[o.bucket_label] = by_bucket.get(o.bucket_label, 0) + 1

    print()
    print(f"=== {label} ===")
    print(f"total:   {total}")
    print(f"pass:    {pass_n}")
    print(f"timeout: {timeout_n}")
    print(f"fail:    {fail_n}")
    if by_exc:
        print(f"failures by exception class:")
        for exc, n in sorted(by_exc.items(), key=lambda kv: -kv[1]):
            print(f"  {n:3d}  {exc}")
    return {
        "label": label,
        "total": total,
        "pass": pass_n,
        "timeout": timeout_n,
        "fail": fail_n,
        "by_exception": by_exc,
        "outcomes": {o.name: {"status": o.status, "detail": o.detail,
                              "elapsed": o.elapsed, "raw": o.raw}
                     for o in outcomes},
    }


def delta(before: dict, after: dict) -> None:
    """Print benchmarks that moved between buckets."""
    print()
    print("=== DELTA (off → on) ===")
    b_outs = before["outcomes"]
    a_outs = after["outcomes"]
    moved_to_pass: list[tuple[str, str]] = []
    regressed: list[tuple[str, str, str]] = []
    other: list[tuple[str, str, str]] = []
    for name in sorted(set(b_outs) | set(a_outs)):
        b_st = b_outs.get(name, {}).get("status", "MISSING")
        a_st = a_outs.get(name, {}).get("status", "MISSING")
        b_dt = b_outs.get(name, {}).get("detail", "")
        a_dt = a_outs.get(name, {}).get("detail", "")
        if b_st == a_st and b_dt == a_dt:
            continue
        if b_st != "PASS" and a_st == "PASS":
            moved_to_pass.append((name, f"{b_st}:{b_dt}" if b_dt else b_st))
        elif b_st == "PASS" and a_st != "PASS":
            regressed.append((name, b_st, f"{a_st}:{a_dt}" if a_dt else a_st))
        else:
            other.append((name, f"{b_st}:{b_dt}" if b_dt else b_st,
                          f"{a_st}:{a_dt}" if a_dt else a_st))
    print(f"new wins (off→on, fail→pass): {len(moved_to_pass)}")
    for name, prev in moved_to_pass:
        print(f"  + {name:<60s}  was: {prev}")
    print(f"regressions (off→on, pass→fail/timeout): {len(regressed)}")
    for name, b, a in regressed:
        print(f"  ! {name:<60s}  was: {b}  now: {a}")
    print(f"other movement (e.g. fail-class change): {len(other)}")
    for name, b, a in other:
        print(f"    {name:<60s}  {b}  →  {a}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--smt", choices=["on", "off", "both"], default="on")
    p.add_argument("--filter", default=None)
    p.add_argument("--out", default=None,
                   help="JSON output path. For --smt both, suffixes "
                        "`.off.json` and `.on.json` are appended.")
    p.add_argument(
        "--workers", type=int,
        default=max(1, (os.cpu_count() or 2) // 2),
    )
    args = p.parse_args()

    benches = discover_benchmarks(args.filter)
    print(f"discovered {len(benches)} benchmarks under benchmarking/*/*.py")
    print(f"workers: {args.workers}, timeout: {args.timeout}s, smt: {args.smt}")
    print(f"interpreter: {PYTHON}")

    results: dict[str, dict] = {}

    def run(label: str, smt_on: bool) -> dict:
        print()
        print(f"--- sweep [{label}] ---")
        t0 = time.time()
        outs = sweep(benches, args.timeout, smt_on, args.workers)
        wall = time.time() - t0
        print(f"wall-clock: {wall:.1f}s")
        summary = summarise(label, outs)
        summary["wall_clock_s"] = wall
        if args.out:
            path = (Path(args.out).with_suffix(f".{label}.json")
                    if args.smt == "both" else Path(args.out))
            path.write_text(json.dumps(summary, indent=2, default=str))
            print(f"wrote {path}")
        return summary

    if args.smt in ("off", "both"):
        results["off"] = run("off", smt_on=False)
    if args.smt in ("on", "both"):
        results["on"] = run("on", smt_on=True)

    if args.smt == "both":
        delta(results["off"], results["on"])
        # Net change
        net = results["on"]["pass"] - results["off"]["pass"]
        print()
        print(f"NET CHANGE: pass {results['off']['pass']} -> "
              f"{results['on']['pass']} ({net:+d}) of {results['on']['total']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
