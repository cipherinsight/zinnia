#!/usr/bin/env python3
"""Output-binding audit: mock publics vs numpy ground truth (plan Step 0b).

For every DS-1000 case with an input builder:

  1. Ground truth — run the kernel body as PLAIN numpy (the ``@zk_circuit``
     decorator replaced by an identity shim) on the ``INPUT_BUILDERS``
     fixture, giving the expected output values and runtime shape.
  2. Circuit — compile and ``prove(backend="mock")`` on the same inputs;
     the mock prover honestly evaluates the IR and reports every exposed
     public as ``name=signed-decimal`` (``public_f_*`` are Q32 fixed-point).
  3. Accounting — every expected output element must appear among the
     publics (exact for integers, fixed-point tolerance for floats), and
     every public NOT consumed that way must be explainable as padding
     (zero) or dyn-array metadata (a runtime dim / length / total of the
     expected output). An empty or truncated payload therefore fails even
     when the values that ARE present all match.

This is the systematic version of the spot checks that caught the
0304/0507/0509/0510 empty-payload bindings and the 0415/0416/0418
wrong-reshape bindings.

Usage:
    python audit_bindings.py [--cases 0301,0415 | all] [--jobs N]
                             [--output audit_bindings.json]

Exit status 0 iff all audited cases pass.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(THIS_DIR.parent))

os.environ.setdefault("ZINNIA_SKIP_RELEASE_BUILD", "1")

from run_halo2_baselines import INPUT_BUILDERS  # noqa: E402
from run_zinnia_noir import _coerce_arg  # noqa: E402

Q = 1 << 32  # fpc fixed-point scale (precision_bits=32)
# Float matching tolerance, in REAL units. The mock backend evaluates with
# Q32 fixed-point semantics (rounded multiplies, Remez transcendentals), so
# numpy float64 ground truth drifts by ULPs-to-1e-5 on op chains. Binding
# bugs are gross (missing/zero/permuted payloads), so a loose-ish tolerance
# loses no detection power.
FLOAT_ATOL = 1e-3
FLOAT_RTOL = 1e-4


def _identity_zk_circuit(method=None, **_kwargs):
    if method is None:
        return lambda m: m
    return method


# Ground-truth overrides for kernels whose bodies are not executable as
# plain numpy (Zinnia-specific semantics). Keyed by case; receives the
# coerced positional args and must return what the kernel SHOULD return.
def _gt_0344(mystr):
    import numpy as np
    # Python `list(int)` is a TypeError; Zinnia defines `list(Integer)` as
    # the single-element list `[Integer]` (the DS-1000 original took a
    # STRING and split digits — unrepresentable without string inputs, an
    # accepted semantic divergence of the acceptance study). Ground truth
    # follows the Zinnia semantics the kernel actually has.
    return np.array([int(mystr)], dtype=int)


GROUND_TRUTH_OVERRIDES = {
    "0344": _gt_0344,
}

# Documented under-bindings deferred to the provenance plan's Step 3
# (slice-view provenance with runtime extents — the "encoding fork").
# Dynamic-bounds slices of STATIC arrays currently return a full-envelope
# StaticArray: payload values are correct and verified, but the runtime
# bbox shape is not pinned by the contract (the if-merge then surfaces
# static envelope dims as metadata). Step 3's verification criteria
# include deleting these entries.
EXPECTED_FAILURES = {
    "0507": "bbox slice metadata = static envelope dims (plan Step 3)",
    "0509": "bbox slice metadata = static envelope dims (plan Step 3)",
    "0510": "bbox slice metadata = static envelope dims (plan Step 3)",
}


def numpy_ground_truth(case: str, args: list):
    """Run the kernel body as plain numpy. Returns the raw return value."""
    import numpy as np
    import zinnia as _zinnia

    kernel_path = THIS_DIR / f"ds1000_case{case}" / "kernel.py"
    src = kernel_path.read_text()
    ns = {name: getattr(_zinnia, name) for name in dir(_zinnia) if not name.startswith("_")}
    ns["np"] = np
    ns["zk_circuit"] = _identity_zk_circuit
    # Chips/externals are plain helper functions under numpy semantics.
    ns["zk_chip"] = _identity_zk_circuit
    ns["zk_external"] = _identity_zk_circuit
    # Drop the real import so the shim above survives module execution.
    src = src.replace("from zinnia import *", "")
    exec(compile(src, str(kernel_path), "exec"), ns)
    call_args = [np.array(a, copy=True) if isinstance(a, np.ndarray) else a for a in args]
    return ns["kernel"](*call_args)


def flatten_expected(result):
    """Flatten a kernel return value into (int_values, float_values, meta).

    meta is the set of integers that legitimately appear as dyn-array
    metadata publics for this output: runtime dims, lengths, totals.
    """
    import numpy as np

    # Tuples AND lists of heterogeneous-shape arrays (e.g. ragged window
    # lists from chips) flatten part-by-part.
    if isinstance(result, tuple):
        parts = result
    elif isinstance(result, list) and any(isinstance(p, np.ndarray) for p in result):
        parts = tuple(result)
    else:
        parts = (result,)
    exp_ints: list[int] = []
    exp_floats: list[float] = []
    allowed_meta: set[int] = {0}
    for part in parts:
        arr = np.asarray(part)
        allowed_meta.add(int(arr.size))
        for d in arr.shape:
            allowed_meta.add(int(d))
        if arr.ndim >= 1:
            # products of trailing dims (length-style metadata for 2-D dyn
            # encodings: rows*cols, etc.)
            total = 1
            for d in arr.shape:
                total *= int(d)
            allowed_meta.add(total)
        flat = arr.ravel()
        if arr.dtype.kind in "fc":
            exp_floats.extend(float(v) for v in flat)
        elif arr.dtype.kind == "b":
            exp_ints.extend(int(v) for v in flat)
        else:
            exp_ints.extend(int(v) for v in flat)
    return exp_ints, exp_floats, allowed_meta


def parse_publics(public_values: list[str]):
    """Split mock publics into int and float value lists (signed decimals)."""
    pub_ints: list[int] = []
    pub_floats: list[float] = []
    for entry in public_values:
        name, _, val = entry.partition("=")
        v = int(val)
        if name.startswith("public_f_"):
            pub_floats.append(v / Q)
        else:
            pub_ints.append(v)
    return pub_ints, pub_floats


def _match_floats(expected: list[float], available: list[float]):
    """Greedy nearest matching on sorted lists. Returns (missing, leftover)."""
    exp = sorted(expected)
    avail = sorted(available)
    used = [False] * len(avail)
    missing = []
    i = 0
    for e in exp:
        tol = max(FLOAT_ATOL, FLOAT_RTOL * abs(e))
        # advance to the first unused candidate >= e - tol
        while i < len(avail) and (used[i] or avail[i] < e - tol):
            i += 1
        best = None
        for j in range(i, len(avail)):
            if used[j]:
                continue
            if avail[j] > e + tol:
                break
            if best is None or abs(avail[j] - e) < abs(avail[best] - e):
                best = j
        if best is None:
            missing.append(e)
        else:
            used[best] = True
    leftover = [avail[j] for j in range(len(avail)) if not used[j]]
    return missing, leftover


def audit_one(case: str) -> dict:
    import numpy as np

    rec: dict = {"case": case, "ok": False}
    t0 = time.monotonic()
    try:
        from zinnia.api.zk_circuit import ZKCircuit

        kernel_path = THIS_DIR / f"ds1000_case{case}" / "kernel.py"
        spec = importlib.util.spec_from_file_location(f"audit_case{case}", kernel_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        compiled = ZKCircuit.from_method(mod.kernel).compile()

        input_dict = INPUT_BUILDERS[case]()

        def lookup(name: str):
            for key in (name, name.lower(), name.upper(),
                        f"{name}_val", f"{name.lower()}_val"):
                if key in input_dict:
                    return input_dict[key]
            raise KeyError(f"no builder key for kernel param {name!r}")

        args = [_coerce_arg(lookup(pi.name), pi.export().get("dt"))
                for pi in compiled.program_inputs]

        if case in GROUND_TRUTH_OVERRIDES:
            expected_raw = GROUND_TRUTH_OVERRIDES[case](*args)
        else:
            expected_raw = numpy_ground_truth(case, args)
        exp_ints, exp_floats, allowed_meta = flatten_expected(expected_raw)
        rec["expected_elements"] = len(exp_ints) + len(exp_floats)

        proof = compiled.prove(*args, backend="mock")
        if proof.proof_bytes_hex != "mock_satisfied":
            rec["error"] = "mock prove unsatisfied on honest witness"
            return rec
        pub_ints, pub_floats = parse_publics(proof.public_values)
        # Constant output cells the compiler elided from the instance are
        # still part of the output contract (pinned by the kernel text);
        # the compiler reports its claim and we audit it like a public.
        elided = compiled.get_elided_publics()
        for entry in elided:
            if "Int" in entry:
                pub_ints.append(int(entry["Int"]))
            else:
                pub_floats.append(float(entry["Float"]))
        rec["publics"] = len(pub_ints) + len(pub_floats)
        rec["elided"] = len(elided)

        problems: list[str] = []

        # Integer payload: exact multiset containment.
        pub_int_counter = Counter(pub_ints)
        missing_ints = []
        for v in exp_ints:
            if pub_int_counter[v] > 0:
                pub_int_counter[v] -= 1
            else:
                missing_ints.append(v)
        leftover_ints = list(pub_int_counter.elements())

        # Float payload: tolerance multiset containment.
        missing_floats, leftover_floats = _match_floats(exp_floats, pub_floats)

        # Cross-pool rescue: int/float placement can legitimately differ
        # (e.g. an integral-valued float output exposed as public_i).
        if missing_ints and leftover_floats:
            still_missing, leftover_floats = _match_floats(
                [float(v) for v in missing_ints], leftover_floats)
            missing_ints = [int(v) for v in still_missing]
        if missing_floats and leftover_ints:
            rescued, kept_missing = [], []
            li = Counter(leftover_ints)
            for e in missing_floats:
                r = round(e)
                if abs(e - r) <= max(FLOAT_ATOL, FLOAT_RTOL * abs(e)) and li[r] > 0:
                    li[r] -= 1
                    rescued.append(e)
                else:
                    kept_missing.append(e)
            missing_floats = kept_missing
            leftover_ints = list(li.elements())

        if missing_ints:
            problems.append(f"missing int outputs: {missing_ints[:8]}"
                            + ("..." if len(missing_ints) > 8 else ""))
        if missing_floats:
            problems.append(f"missing float outputs: {missing_floats[:8]}"
                            + ("..." if len(missing_floats) > 8 else ""))

        # Remainder accounting: leftovers must be padding or metadata.
        bad_ints = [v for v in leftover_ints if v not in allowed_meta]
        bad_floats = [v for v in leftover_floats if abs(v) > FLOAT_ATOL]
        if bad_ints:
            problems.append(f"unexplained int publics: {sorted(set(bad_ints))[:8]}")
        if bad_floats:
            problems.append(f"unexplained float publics: {bad_floats[:8]}")

        # Payload-size sanity: publics must at least carry every expected
        # element (catches truncated envelopes even when zeros dominate).
        if rec["publics"] < rec["expected_elements"]:
            problems.append(
                f"public count {rec['publics']} < expected element count "
                f"{rec['expected_elements']} (truncated payload)")

        if problems:
            rec["error"] = "; ".join(problems)
        else:
            rec["ok"] = True
    except BaseException as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc(limit=6)
    finally:
        rec["seconds"] = round(time.monotonic() - t0, 2)
    return rec


def _worker(case: str, queue: mp.Queue):
    queue.put(audit_one(case))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="all",
                    help="comma-separated case ids, or 'all'")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--output", default=str(THIS_DIR / "audit_bindings.json"))
    args = ap.parse_args()

    if args.cases == "all":
        cases = sorted(INPUT_BUILDERS)
    else:
        cases = [c.strip() for c in args.cases.split(",") if c.strip()]

    ctx = mp.get_context("fork")
    pending = list(cases)
    running: dict[str, tuple] = {}
    results: list[dict] = []

    while pending or running:
        while pending and len(running) < args.jobs:
            case = pending.pop(0)
            q = ctx.Queue()
            p = ctx.Process(target=_worker, args=(case, q))
            p.start()
            running[case] = (p, q, time.monotonic())
        time.sleep(0.05)
        for case in list(running):
            p, q, started = running[case]
            rec = None
            if not q.empty():
                rec = q.get()
            elif not p.is_alive():
                rec = {"case": case, "ok": False,
                       "error": f"worker died (exit {p.exitcode})"}
            elif time.monotonic() - started > args.timeout:
                p.terminate()
                rec = {"case": case, "ok": False, "error": "timeout"}
            if rec is not None:
                p.join(timeout=5)
                del running[case]
                results.append(rec)
                status = "ok" if rec["ok"] else f"FAIL  {rec.get('error', '')[:120]}"
                print(f"[{len(results):3}/{len(cases)}] {case}: {status}", flush=True)

    results.sort(key=lambda r: r["case"])
    failures = [r for r in results if not r["ok"]]
    unexpected = [r for r in failures if r["case"] not in EXPECTED_FAILURES]
    expected = [r for r in failures if r["case"] in EXPECTED_FAILURES]
    fixed_xfails = [c for c in EXPECTED_FAILURES
                    if any(r["case"] == c and r["ok"] for r in results)]
    summary = {
        "total": len(results),
        "passed": len(results) - len(failures),
        "failed": len(failures),
        "unexpected_failures": [
            {"case": r["case"], "error": r.get("error")} for r in unexpected
        ],
        "expected_failures": [
            {"case": r["case"], "reason": EXPECTED_FAILURES[r["case"]],
             "error": r.get("error")} for r in expected
        ],
        "results": results,
    }
    Path(args.output).write_text(json.dumps(summary, indent=2))
    print(f"\n{summary['passed']}/{summary['total']} cases pass binding audit "
          f"-> {args.output}")
    if expected:
        print("XFAIL (deferred, see EXPECTED_FAILURES):",
              " ".join(r["case"] for r in expected))
    if fixed_xfails:
        print("XPASS — remove from EXPECTED_FAILURES:", " ".join(fixed_xfails))
    if unexpected:
        print("FAILING:", " ".join(r["case"] for r in unexpected))
    sys.exit(1 if unexpected else 0)


if __name__ == "__main__":
    main()
