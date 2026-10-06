#!/usr/bin/env python3
"""Differential test: Zinnia's Noir backend against the mock backend.

For every `perf` case of `datasets/case_sets/ds1000_perf.json` and both inputs
of its `fixture.json` (A and B):

  1. compile the kernel (`harness.kernels.compile_kernel`) and take the input's
     arguments from the fixture (`harness.fixtures`);
  2. mock: `prove(backend="mock")`; its public values, ordered by statement
     id (the order in which the IR exposes them);
  3. Noir: emit the circuit and run `nargo execute` (`execute_noir_circuit`),
     which returns what `main` returns, in the same IR order. With `--prove`
     the case is instead proved with `bb` and verified, and the compared
     values are the public inputs bb proved (`ProofArtifact.public_values`);
  4. compare position by position. Integer outputs (`public_i_*`) must be
     equal. Float outputs (`public_f_*`, Q32 integers) must be equal too
     (sqrt, sin, cos, tan and arcsin included: the mock and noir_fpc run the
     same fixed-point algorithms of `src/prove/kernel.rs`), unless the kernel
     uses a function the mock evaluates in f64 (pow, exp, log, sinh, cosh,
     tanh, arccos, arctan2): those kernels get the tolerance `--approx-rtol` /
     `--approx-atol` (real units) on every output.

A case's status is the worse of its two inputs. Each case runs in a child
process with a timeout; `--wrap` prefixes every child command, e.g. with the
memory cap (a case that exceeds it is then recorded as a crash instead of
killing the sweep):

    MEMRUN_MAX=8G .venv/bin/python benchmarking/noir_differential.py \\
        --wrap ../eval-redo/tools/memrun.sh [--cases 0303,0353] [--prove] \\
        [--jobs 2] [--out noir_differential.json]

Exit status 0 iff no case mismatches and no case fails on the Noir side
while the mock succeeds.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
DATASET = THIS_DIR / "datasets" / "ds1000"
CASE_SET = THIS_DIR / "datasets" / "case_sets" / "ds1000_perf.json"
INPUTS = ("A", "B")
os.environ.setdefault("ZINNIA_SKIP_RELEASE_BUILD", "1")

BN254_P = 21888242871839275222246405745257275088548364400416034343698204186575808495617
Q = 1 << 32
PARAMS = {"k": 0, "precision_bits": 32, "lookup_bits": 8}

# IR ops whose mock semantics is f64 math rather than exact fixed point.
APPROX_OPS = {
    "SinHFIR", "CosHFIR", "TanHFIR", "ExpFIR", "LogFIR", "PowFIR", "PowFIIR",
    "ArcCosFIR", "ArcTan2FIR",
}

# Statuses from best to worst; a case reports the worst of its inputs.
STATUS_ORDER = ("match", "match_approx", "both_reject", "compile_error", "mock_error",
                "mismatch", "noir_error", "mock_unsatisfied_noir_accepts")
FAILING = {"mismatch", "noir_error", "mock_unsatisfied_noir_accepts", "crash", "timeout"}


def signed(v: int) -> int:
    """Signed reading of a field element (as the mock's fp_to_i64)."""
    v %= BN254_P
    return v - BN254_P if v > BN254_P // 2 else v


def case_name(spec: str) -> str:
    return spec if spec.startswith("ds1000_case") else f"ds1000_case{spec}"


def perf_cases() -> list[str]:
    """Every case whose status in the performance case set is `perf`."""
    entries = json.loads(CASE_SET.read_text())["cases"]
    return sorted(e["case"] for e in entries if e["status"] == "perf")


def load_case(case: str):
    """(compiled program, fixture) of one DS-1000 case folder."""
    sys.path.insert(0, str(ROOT))
    from benchmarking.harness.fixtures import load_fixture
    from benchmarking.harness.kernels import compile_kernel

    directory = DATASET / case_name(case)
    fixture = load_fixture(directory / "fixture.json")
    return compile_kernel(directory / "kernel.py", fixture.function, backend="noir"), fixture


def mock_publics(compiled, args) -> tuple[bool, list[tuple[str, int]]]:
    """(satisfied, [(kind, value)]) with kind 'i' or 'f', in statement order."""
    proof = compiled.prove(*args, backend="mock", params=PARAMS)
    entries = []
    for pv in proof.public_values:
        label, value = pv.split("=", 1)
        m = re.fullmatch(r"public_([if])_(\d+)", label)
        if m is None:
            raise ValueError(f"unexpected mock public label {label!r}")
        entries.append((int(m.group(2)), m.group(1), int(value)))
    entries.sort()
    return proof.proof_bytes_hex == "mock_satisfied", [(k, v) for _, k, v in entries]


def noir_execute(compiled, args) -> list[int]:
    from zinnia._zinnia_core import execute_noir_circuit
    from zinnia.exec.input_parser import build_circuit_inputs

    witness = build_circuit_inputs(compiled.program_inputs, args)
    returned = execute_noir_circuit(
        compiled._compiled_ir, json.dumps(witness), {}, json.dumps(PARAMS)
    )
    return [signed(int(v, 16)) for v in json.loads(returned)]


def noir_prove(compiled, args, n_outputs: int) -> list[int]:
    proof = compiled.prove(*args, backend="noir", params=PARAMS)
    if not compiled.verify(proof):
        raise RuntimeError("bb verify rejected the Noir proof")
    values = [signed(int(v, 16)) for v in proof.public_values]
    # Public inputs come first in bb's list; the outputs are the tail.
    return values[len(values) - n_outputs:] if n_outputs else []


def compare(mock: list[tuple[str, int]], noir: list[int], approx: bool,
            rtol: float, atol: float) -> dict:
    out = {"n_outputs": len(mock), "n_noir": len(noir)}
    if len(mock) != len(noir):
        out["status"] = "mismatch"
        out["reason"] = f"output count differs: mock {len(mock)}, noir {len(noir)}"
        return out
    bad = []
    max_ulps = 0
    for pos, ((kind, m), n) in enumerate(zip(mock, noir)):
        diff = abs(m - n)
        if kind == "f":
            max_ulps = max(max_ulps, diff)
        if diff == 0:
            continue
        if approx:
            scale = Q if kind == "f" else 1
            if diff / scale <= atol + rtol * abs(m) / scale:
                continue
        bad.append({"pos": pos, "kind": kind, "mock": m, "noir": n})
    out["max_float_diff_ulps"] = max_ulps
    out["status"] = "mismatch" if bad else ("match_approx" if approx and max_ulps else "match")
    if bad:
        out["n_bad"] = len(bad)
        out["first_bad"] = bad[:8]
    return out


def run_input(compiled, args, approx: bool, prove: bool, rtol: float, atol: float) -> dict:
    rec: dict = {}
    try:
        satisfied, mock = mock_publics(compiled, args)
    except BaseException as e:  # noqa: BLE001 - PyO3 panics derive from BaseException
        return {"status": "mock_error", "error": f"{type(e).__name__}: {e}"[:2000]}
    rec["mock_satisfied"] = satisfied
    try:
        noir = noir_prove(compiled, args, len(mock)) if prove else noir_execute(compiled, args)
    except BaseException as e:  # noqa: BLE001
        status = "noir_error" if satisfied else "both_reject"
        rec.update(status=status, error=f"{type(e).__name__}: {e}"[-2000:])
        return rec
    if not satisfied:
        rec.update(status="mock_unsatisfied_noir_accepts")
        return rec
    rec.update(compare(mock, noir, approx, rtol, atol))
    return rec


def run_child(case: str, prove: bool, rtol: float, atol: float) -> dict:
    rec: dict = {"case": case_name(case)}
    t0 = time.time()
    try:
        compiled, fixture = load_case(case)
    except BaseException as e:  # noqa: BLE001
        rec.update(status="compile_error", error=f"{type(e).__name__}: {e}"[:2000])
        return rec
    ops = {s["ir_instance"]["__class__"] for s in compiled.get_ir_stmts()}
    approx = sorted(ops & APPROX_OPS)
    rec["approx_ops"] = approx
    rec["inputs"] = {name: run_input(compiled, fixture.arguments(name), bool(approx), prove,
                                     rtol, atol) for name in INPUTS}
    rec["status"] = max((r["status"] for r in rec["inputs"].values()), key=STATUS_ORDER.index)
    rec["seconds"] = round(time.time() - t0, 2)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cases", default="all", help="comma-separated case numbers, or 'all'")
    ap.add_argument("--prove", action="store_true", help="prove with bb instead of nargo execute")
    ap.add_argument("--timeout", type=int, default=900, help="per-case seconds")
    ap.add_argument("--approx-rtol", type=float, default=1e-6)
    ap.add_argument("--approx-atol", type=float, default=1e-6)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default=str(THIS_DIR / "noir_differential.json"))
    ap.add_argument("--wrap", default="", help="command prefix for each child (e.g. a memory cap)")
    ap.add_argument("--child", help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a.child:
        print(json.dumps(run_child(a.child, a.prove, a.approx_rtol, a.approx_atol)))
        return 0

    cases = perf_cases() if a.cases == "all" else [case_name(c) for c in a.cases.split(",")]

    def run(case: str) -> dict:
        cmd = a.wrap.split() + [sys.executable, __file__, "--child", case,
                                "--approx-rtol", str(a.approx_rtol),
                                "--approx-atol", str(a.approx_atol)]
        if a.prove:
            cmd.append("--prove")
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout)
            lines = [l for l in proc.stdout.splitlines() if l.startswith("{")]
            return json.loads(lines[-1]) if lines else {
                "case": case, "status": "crash", "error": proc.stderr[-2000:]}
        except subprocess.TimeoutExpired:
            return {"case": case, "status": "timeout"}

    results = []
    with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as pool:
        for rec in pool.map(run, cases):
            results.append(rec)
            detail = ""
            if rec["status"] not in ("match", "match_approx"):
                bad = [r for r in rec.get("inputs", {}).values()
                       if r["status"] not in ("match", "match_approx")]
                first = bad[0] if bad else rec
                detail = f" ({first.get('reason') or first.get('error', '')[:200]})"
            print(f"{rec['case']}: {rec['status']}{detail}", flush=True)
            Path(a.out).write_text(json.dumps(results, indent=1))

    counts: dict[str, int] = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print("summary:", json.dumps(counts, sort_keys=True))
    return 1 if any(r["status"] in FAILING for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
