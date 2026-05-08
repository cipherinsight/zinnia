#!/usr/bin/env python3
"""
Profile a single benchmark with SMT telemetry on. Captures wall-clock
compile time + the SmtTelemetry summary (which the Rust side dumps to
stderr when ZINNIA_SMT_LOG_TELEMETRY=1) and writes both to a `.txt` file
under tools/sweep_results/.

Usage:
    python tools/profile_one.py benchmarking/guerre/guerre.py [--timeout 1800]

The benchmark runs in a child process (same DRIVER as sweep_benchmarks.py)
so a crash can't take down the profiling driver. Output path is
`tools/sweep_results/profile_<benchmark_stem>.txt`.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROFILE_DIR = REPO_ROOT / "tools" / "sweep_results"


def _venv_python() -> str:
    candidate = REPO_ROOT / ".venv" / "bin" / "python"
    if candidate.exists():
        return str(candidate)
    return sys.executable


PYTHON = _venv_python()


# Same minimal driver as sweep_benchmarks.py — find @zk_circuit, compile.
DRIVER = r"""
import json, sys, importlib.util, os

bench_path = sys.argv[1]
bench_dir = os.path.dirname(bench_path)
bench_stem = os.path.splitext(os.path.basename(bench_path))[0]
sys.path.insert(0, bench_dir)

t0 = __import__('time').time()
spec = importlib.util.spec_from_file_location(bench_stem, bench_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

from zinnia.api.zk_circuit import ZKCircuit
candidates = []
for name, obj in vars(mod).items():
    if name.startswith("_"):
        continue
    if callable(obj) and getattr(obj, "__name__", None) == "__zk_circuit_annotator_inner":
        candidates.append((name, obj))
if not candidates:
    print("NO_CIRCUIT", file=sys.stderr)
    sys.exit(2)

for name, fn in candidates:
    t1 = __import__('time').time()
    circuit = ZKCircuit.from_method(fn)
    circuit.compile()
    sys.stderr.write(f"  compiled {name} in {__import__('time').time()-t1:.2f}s\n")

print(f"PASS elapsed={__import__('time').time()-t0:.2f}s")
"""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("bench")
    p.add_argument("--timeout", type=float, default=1800.0)
    p.add_argument("--smt-timeout-ms", type=int, default=None,
                   help="Override ZINNIA_SMT_QUERY_TIMEOUT_MS for this run.")
    p.add_argument("--out", default=None,
                   help="Output path. Defaults to "
                        "tools/sweep_results/profile_<stem>.txt")
    args = p.parse_args()

    bench = Path(args.bench)
    if not bench.is_absolute():
        bench = (REPO_ROOT / bench).resolve()
    if not bench.exists():
        print(f"benchmark not found: {bench}", file=sys.stderr)
        return 1
    stem = bench.parent.name + "_" + bench.stem
    out = Path(args.out) if args.out else PROFILE_DIR / f"profile_{stem}.txt"

    env = os.environ.copy()
    env["ZINNIA_SMT_ENABLE"] = "1"
    env["ZINNIA_SMT_LOG_TELEMETRY"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if args.smt_timeout_ms is not None:
        env["ZINNIA_SMT_QUERY_TIMEOUT_MS"] = str(args.smt_timeout_ms)

    print(f"profiling: {bench}", file=sys.stderr)
    print(f"  ZINNIA_SMT_QUERY_TIMEOUT_MS={env.get('ZINNIA_SMT_QUERY_TIMEOUT_MS', '500 (default)')}",
          file=sys.stderr)
    t0 = time.time()
    try:
        proc = subprocess.run(
            [PYTHON, "-c", DRIVER, str(bench)],
            capture_output=True,
            text=True,
            timeout=args.timeout,
            cwd=str(REPO_ROOT),
            env=env,
        )
        timed_out = False
    except subprocess.TimeoutExpired as e:
        proc = e
        timed_out = True
    elapsed = time.time() - t0

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write(f"benchmark: {bench.relative_to(REPO_ROOT)}\n")
        f.write(f"timed_out: {timed_out}\n")
        f.write(f"wall_clock_s: {elapsed:.2f}\n")
        f.write(f"smt_timeout_ms_env: {env.get('ZINNIA_SMT_QUERY_TIMEOUT_MS', '(unset, default=500)')}\n")
        f.write("--- stdout ---\n")
        f.write(getattr(proc, "stdout", "") or "")
        f.write("\n--- stderr ---\n")
        f.write(getattr(proc, "stderr", "") or "")

    print(f"wrote {out} (elapsed {elapsed:.1f}s, timed_out={timed_out})", file=sys.stderr)
    return 0 if not timed_out else 124


if __name__ == "__main__":
    sys.exit(main())
