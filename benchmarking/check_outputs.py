#!/usr/bin/env python3
"""Position-aware output check: Zinnia's circuit outputs vs NumPy, by the output layout.

For every DS-1000 case with a fixture (`datasets/ds1000/<case>/fixture.json`;
the performance set's cases by default):

1. NumPy reference: the kernel body run as plain NumPy on the fixture input
   (`harness/reference.py`; no per-case overrides). A kernel NumPy rejects is
   reported as `numpy_error`.
2. Circuit outputs: compile the kernel, prove it on the mock backend and
   decode the public values with the compiler's output layout
   (`ZKCompiledProgram.get_output_layout`, `harness/outputs.py`): every public
   value is a known output element or shape entry, every constant the compiler
   left out of the public vector sits at its known position, and a
   runtime-shaped result is read at its runtime shape (padding is not data).
3. Comparison (`harness/outputs.check_outputs`): element by element in output
   order, shapes equal, integers exactly, floats within the case's tolerance
   (`float_tolerance` in the case set, else the harness default).
4. Tampered-public controls, per case and input: the same decode and
   comparison on a copy of the public vector with one output element changed,
   with two distinguishable output elements swapped, and with one exposed
   shape entry changed. Each must be rejected (a decode error or a mismatch);
   otherwise the check has no power on that case and the case is flagged
   `weak_control`.

This replaces fix-core's layout-guessing version (which matched the outputs
against every plausible layout) now that the compiler exports its layout.

Usage:
    python -m benchmarking.check_outputs [--cases 0301,0415] [--status perf|all]
        [--inputs A,B] [--output FILE] [--timeout 600] [--memory 8G] [--backend halo2|noir]

`--backend` is the compile target (`ZinniaConfig.backend`); it selects between
equivalent lowerings (a sort's network or pinned hints), so both targets'
circuits are checked.

Each case runs in its own process under the memory cap. Exit status 0 iff
every case passes. Statuses: `pass`; `pass_constant_output` (every output
value is a compile-time constant or the result is empty, so there is no public
value to tamper with); `mismatch`, `layout_error`, `unsatisfied` (the honest
input violates the circuit), `numpy_error`, `compile_error`, `weak_control`,
`timeout`, `crashed`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarking.harness import paths  # noqa: E402
from benchmarking.harness.case_sets import DEFAULT_TOLERANCE, parse_tolerance  # noqa: E402
from benchmarking.harness.isolation import memory_capped  # noqa: E402

PERF_CASE_SET = paths.BENCHMARKING / "datasets" / "case_sets" / "ds1000_perf.json"


# ── One case (child process) ─────────────────────────────────────────────

def mock_exposed_values(proof) -> list[int]:
    """The mock's public values in exposure order (signed; floats in Q32).

    The mock names each exposed public after its statement id; sorting by the
    id restores the exposure order a real backend uses."""
    entries = []
    for entry in proof.public_values:
        name, _, value = entry.partition("=")
        entries.append((int(name.rsplit("_", 1)[1]), int(value)))
    return [value for _, value in sorted(entries)]


def element_publics(layout: list[dict]) -> tuple[list[tuple[int, str]], list[int]]:
    """(public index, kind) of every exposed output element, and the public
    indices of exposed shape entries."""
    elements, metadata = [], []
    for part in layout:
        if "dynamic" in part:
            dynamic = part["dynamic"]
            cells = dynamic["cells"]
            meta_slots = [dynamic["length"], dynamic["rank"], *dynamic["dims"]]
            metadata.extend(slot["public"] for slot in meta_slots if "public" in slot)
        else:
            cells = part["cells"]
        elements.extend((slot["public"], slot["kind"]) for slot in cells if "public" in slot)
    return elements, metadata


def tampered_vectors(fields: list, elements: list[tuple[int, str]], metadata: list[int], decode,
                     tolerance) -> list[tuple[str, list]]:
    """Controls: one output element changed; two distinguishable output
    elements swapped; one exposed shape entry changed by one.

    An element is changed by one (an integer) or by `1 + 2 * tolerance` (a
    float), so the change is never within the tolerance. Padding cells of
    runtime-shaped results are public too but are not data, so only cells the
    honest decode reads count: each candidate is changed alone and kept if
    the decoded values move. Two elements are swapped only when every decoded
    value the swap moves moves by more than twice the tolerance (a swap the
    tolerance cannot see would make the check look weak by construction)."""
    honest = decode(fields)

    def visible(a, b) -> bool:
        return abs(a - b) > 2 * (tolerance.abs + tolerance.rel * max(abs(a), abs(b)))

    def bumped(index: int, kind: str) -> list:
        changed = list(fields)
        if kind == "f":
            magnitude = abs(changed[index]) / 2**32
            delta = 1 + 2 * (tolerance.abs + tolerance.rel * (magnitude + 2))
            changed[index] += round(delta * 2**32)
        else:
            changed[index] += 1
        return changed

    live = []
    for index, kind in elements:
        try:
            if decode(bumped(index, kind)) != honest:
                live.append((index, kind))
        except ValueError:
            live.append((index, kind))
    controls = []
    if live:
        controls.append(("value_changed", bumped(*live[0])))
        for (i, _), (j, _) in zip(live, live[1:]):
            swapped = list(fields)
            swapped[i], swapped[j] = swapped[j], swapped[i]
            moved = [(a, b) for a, b in zip(decode(swapped), honest) if a != b]
            if moved and all(visible(a, b) for a, b in moved):
                controls.append(("values_swapped", swapped))
                break
    if metadata:
        changed = list(fields)
        changed[metadata[0]] += 1
        controls.append(("shape_changed", changed))
    return controls


def check_case(case: str, inputs: list[str], tolerance_raw, backend: str = "halo2") -> dict:
    from benchmarking.harness.adapters.base import StepFailure
    from benchmarking.harness.adapters.zinnia_common import (describe_compiled,
                                                             with_public_input_gaps,
                                                             zinnia_publics)
    from benchmarking.harness.fixtures import load_fixture
    from benchmarking.harness.kernels import compile_kernel
    from benchmarking.harness.outputs import check_outputs, reference_leaves
    from benchmarking.harness.reference import run_kernel_as_numpy

    tolerance = parse_tolerance(tolerance_raw, "case set") if tolerance_raw else DEFAULT_TOLERANCE
    record: dict = {"case": case, "status": "pass", "tolerance": vars(tolerance), "inputs": {}}
    directory = paths.DS1000_PERF / case
    kernel = directory / "kernel.py"
    fixture = load_fixture(directory / "fixture.json")
    try:
        compiled = compile_kernel(kernel, fixture.function, backend=backend)
    except BaseException as exc:  # noqa: BLE001 - PyO3 panics derive from BaseException
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        record.update(status="compile_error", error=f"{type(exc).__name__}: {str(exc)[:300]}")
        return record
    info = describe_compiled(compiled)
    record["parts"] = len(info["output_layout"])
    elements, metadata = element_publics(info["output_layout"])
    record["exposed_elements"] = len(elements)

    def decode(fields):
        try:
            return zinnia_publics(fields, info).values
        except StepFailure as failure:
            raise ValueError(failure.detail) from None

    statuses = []
    for name in inputs:
        args = fixture.arguments(name)
        result: dict = {}
        record["inputs"][name] = result
        try:
            expected, shapes = reference_leaves(run_kernel_as_numpy(kernel, fixture.function, args))
        except Exception as exc:  # noqa: BLE001
            result.update(status="numpy_error", error=f"{type(exc).__name__}: {exc}"[:300])
            statuses.append("numpy_error")
            continue
        proof = compiled.prove(*args, backend="mock")
        if proof.proof_bytes_hex != "mock_satisfied":
            result.update(status="unsatisfied", error=str(proof.proof_bytes_hex)[:200])
            statuses.append("unsatisfied")
            continue
        fields = with_public_input_gaps(mock_exposed_values(proof), info["public_inputs"])
        try:
            publics = zinnia_publics(fields, info)
        except StepFailure as failure:
            result.update(status="layout_error", error=failure.detail)
            statuses.append("layout_error")
            continue
        check = check_outputs(expected, publics, tolerance, expected_shapes=shapes)
        result["check"] = {"status": check.status, "detail": check.detail,
                           "mismatches": check.mismatches}
        if not check.matched:
            result["status"] = "mismatch"
            statuses.append("mismatch")
            continue
        controls = {}
        for control, tampered in tampered_vectors(fields, elements, metadata, decode,
                                                    tolerance):
            try:
                tampered_check = check_outputs(expected, zinnia_publics(tampered, info),
                                               tolerance, expected_shapes=shapes)
                controls[control] = "ACCEPTED" if tampered_check.matched else "rejected"
            except StepFailure:
                controls[control] = "rejected"
        result["controls"] = controls
        if not controls:
            result["status"] = "pass_constant_output"
        elif any(v != "rejected" for v in controls.values()):
            result["status"] = "weak_control"
        else:
            result["status"] = "pass"
        statuses.append(result["status"])
    order = ["compile_error", "numpy_error", "unsatisfied", "layout_error", "mismatch",
             "weak_control", "pass_constant_output", "pass"]
    record["status"] = min(statuses, key=order.index) if statuses else "pass"
    return record


# ── Driver ───────────────────────────────────────────────────────────────

def selected_cases(case_ids: list[str] | None, status: str) -> list[tuple[str, object]]:
    """(case, tolerance) for the cases to check."""
    entries = json.loads(PERF_CASE_SET.read_text())["cases"]
    by_case = {entry["case"]: entry for entry in entries}
    if case_ids:
        names = [c if c.startswith("ds1000_case") else f"ds1000_case{c}" for c in case_ids]
    else:
        names = [entry["case"] for entry in entries
                 if status == "all" or entry["status"] == status]
    names = [n for n in names if (paths.DS1000_PERF / n / "fixture.json").exists()]
    return [(n, (by_case.get(n) or {}).get("float_tolerance")) for n in names]


def run_isolated(case: str, inputs: list[str], tolerance, timeout: float, memory: str,
                 backend: str) -> dict:
    command = [sys.executable, "-m", "benchmarking.check_outputs", "--one", case,
               "--inputs", ",".join(inputs), "--tolerance", json.dumps(tolerance),
               "--backend", backend]
    started = time.monotonic()
    try:
        proc = subprocess.run(memory_capped(command, memory), capture_output=True, text=True,
                              timeout=timeout, cwd=ROOT, env=dict(os.environ))
    except subprocess.TimeoutExpired:
        return {"case": case, "status": "timeout", "seconds": timeout}
    line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT ")), None)
    record = json.loads(line[7:]) if line else {
        "case": case, "status": "crashed",
        "error": (proc.stderr.strip().splitlines() or [f"exit {proc.returncode}"])[-1][:300]}
    record["seconds"] = round(time.monotonic() - started, 1)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cases", help="comma-separated case ids (default: by --status)")
    parser.add_argument("--status", default="perf",
                        help="case-set status to check (default perf; 'all' = every fixture)")
    parser.add_argument("--inputs", default="A,B", help="fixture inputs to check")
    parser.add_argument("--output", help="write the per-case records to this JSON file")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--memory", default="8G", help="per-case memory cap")
    parser.add_argument("--backend", default="halo2", choices=["halo2", "noir"],
                        help="compile target (selects lowerings such as sorts; default halo2)")
    parser.add_argument("--one", help=argparse.SUPPRESS)
    parser.add_argument("--tolerance", help=argparse.SUPPRESS)
    args = parser.parse_args()
    inputs = [i.strip() for i in args.inputs.split(",") if i.strip()]

    if args.one:
        tolerance = json.loads(args.tolerance) if args.tolerance else None
        print("RESULT " + json.dumps(check_case(args.one, inputs, tolerance, args.backend)), flush=True)
        return

    cases = selected_cases(args.cases.split(",") if args.cases else None, args.status)
    results = []
    for i, (case, tolerance) in enumerate(cases, 1):
        record = run_isolated(case, inputs, tolerance, args.timeout, args.memory, args.backend)
        results.append(record)
        details = {name: r.get("status") for name, r in record.get("inputs", {}).items()}
        print(f"[{i:3}/{len(cases)}] {case}: {record['status']:20} {record.get('error', details)}",
              flush=True)
        if args.output:
            Path(args.output).write_text(json.dumps(results, indent=1))

    counts: dict = {}
    for record in results:
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    print("\n" + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    failing = [r["case"] for r in results if not r["status"].startswith("pass")]
    if failing:
        print("not passing:", " ".join(failing))
    sys.exit(1 if failing else 0)


if __name__ == "__main__":
    main()
