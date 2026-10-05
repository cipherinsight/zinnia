#!/usr/bin/env python3
"""Write `datasets/case_sets/ds1000_perf.json`: one entry per performance-set case
with its status, the reason and the source of the decision, its adaptations,
the output tolerance, and what Zinnia did with it.

    perf_case_set.py --check          # kernels equal the generator's output?
    perf_case_set.py --write          # probe Zinnia, verify fixtures, write the case set

`--write` compiles every in-scope kernel with Zinnia (status `zinnia_rejects`
is provisional and records the commit it was probed at), builds the fixtures
of the cases that need them (`perf_fixtures.py`), mock-proves them, and writes
`fixture.json` for every `perf` case. Set `MEMRUN` to a memory-capping wrapper
(see `zinnia_probe.py`) when running it.
"""

from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ds1000_perf as perf  # noqa: E402
import ds1000_records as records  # noqa: E402
import generate_ds1000_kernels as generator  # noqa: E402
import perf_fixtures as fixtures  # noqa: E402
import zinnia_probe  # noqa: E402

CASE_SET = records.DATASETS / "case_sets" / "ds1000_perf.json"
REPO = records.DATASETS.parents[1]
PREVIOUS_KERNELS = "11cd122"   # bench-reorg: the kernels before this case set
Q = 2 ** 32

STATUS_ORDER = ["out_of_scope", "degenerate", "duplicate", "zinnia_rejects", "perf"]

COPY_TOLERANCE = {"atol": 2.0 ** -32, "rtol": 0.0}
ARITHMETIC_TOLERANCE = {"atol": 1e-6, "rtol": 1e-6}

RULES = {
    "status_order": "A case gets the first status whose rule applies, in this order: " + ", ".join(STATUS_ORDER) + ".",
    "out_of_scope": "The DS-1000 program is out of scope in ds1000_numpy/manifest.json: it imports or uses a module "
                    "other than numpy (incl. submodules) and math (the author's rule), or another rule of "
                    "datasets/FILTERING.md fires (non-numeric data, NaN/infinity, ...), checked on the solution text "
                    "and on every DS-1000 test input.",
    "degenerate": "No output depends on a private input: every output is fixed by the parameter types and the "
                  "constants, so there is nothing to prove.",
    "duplicate": "The kernel computes the same function as a case with a lower id (identical after renaming local "
                 "variables, with the same types and constants); the lower id is kept.",
    "zinnia_rejects": "Zinnia at the commit in `zinnia_probe` refuses the kernel: compilation fails, or it compiles "
                      "but Zinnia refuses to build a witness for either fixture input (error recorded). Provisional: "
                      "compiler fixes on other branches may change it.",
    "perf": "None of the above. These are the cases for the cross-system performance comparison.",
    "kernels": "Every in-scope kernel is the output of tools/generate_ds1000_kernels.py (shapes from the record's "
               "last DS-1000 test case; each dtype the narrowest of Boolean < Integer < Float < Complex holding the "
               "parameter in every test case; returns the value DS-1000 checks) with the shape parameters listed "
               "in tools/ds1000_perf.py bound as compile-time constants. tools/perf_case_set.py --check verifies it.",
    "shape_constants": "Author decision 2026-10-06: in the performance set, DS-1000 shape parameters (values that fix "
                       "an array shape or a trip count) are compile-time constants with their DS-1000 test-case "
                       "value. The runtime variants are in the acceptance set (ds1000_numpy/generated/).",
    "data_movement_only": "true when the outputs depend on private inputs only by copying them (no arithmetic or "
                          "comparison on private values). Such cases stay `perf` but can be reported separately.",
    "numbers": "Every system computes floats in Q32 fixed point (BRIEF.md). Zinnia's quantizer "
               "(src/prove/field.rs `quantize`) rounds v * 2^32 to the nearest integer; BRIEF.md says trunc. "
               "Integer parameters are 64-bit integers.",
    "float_tolerance": "The output check compares each float output with NumPy on input B as "
                       "|actual - expected| <= atol + rtol * |expected|. Outputs that copy an input get atol = 2^-32 "
                       "(one Q32 step), rtol = 0. Computed outputs get atol = rtol = 1e-6: Q32 resolves 2.3e-10 and "
                       "these kernels round at most a few thousand times on values below 10^7, so an implementation "
                       "of the same computation lands well inside, while a precision loss to ~16 fractional bits or "
                       "a wrong value does not. Integer and boolean outputs are compared exactly (`float_tolerance` "
                       "is null).",
}

# Semantic notes for cases whose typed kernel changes what the DS-1000 task is about.
CASE_NOTES: dict[int, str] = {
    317: "The DS-1000 task is converting a nested Python list into an array; with an array parameter "
         "np.array(a) is an identity copy.",
    340: "The DS-1000 task is iterating over every element (the test compares sorted lists); the kernel "
         "returns the C-order list.",
    398: "The fixture draws b from {-1, 1}: DS-1000's randint(2, 10) makes the recurrence leave the Q32 range "
         "within 30 steps.",
    399: "The fixture draws b, c from {-1, 1}: DS-1000's randint(2, 10) makes the recurrence leave the Q32 range "
         "within 30 steps.",
    402: "`a` is unused, as in the DS-1000 reference; dims is its shape.",
    403: "`a` is unused, as in the DS-1000 reference; dims is its shape.",
    420: "x_min and x_max are DS-1000 test inputs (0 and 1 in every test); the previous generator dropped them.",
}


# ---------------------------------------------------------------------------
# Kernel facts
# ---------------------------------------------------------------------------


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout


def previous_kernel(case: str) -> str | None:
    try:
        return git("show", f"{PREVIOUS_KERNELS}:benchmarking/datasets/ds1000/{case}/kernel.py")
    except subprocess.CalledProcessError:
        return None


def _kernel_def(source: str) -> ast.FunctionDef:
    return next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name == "kernel")


def kernel_change(old: str | None, new: str) -> str | None:
    """A one-line description of how the kernel changed, or None if only comments/formatting did."""
    if old is None:
        return "new kernel"
    if perf.kernel_ast(old) == perf.kernel_ast(new):
        return None
    old_def, new_def = _kernel_def(old), _kernel_def(new)
    parts = []
    old_sig = ", ".join(ast.unparse(a) for a in old_def.args.args)
    new_sig = ", ".join(ast.unparse(a) for a in new_def.args.args)
    if old_sig != new_sig:
        parts.append(f"signature ({old_sig}) -> ({new_sig})")
    old_body = [ast.unparse(s) for s in old_def.body]
    new_body = [ast.unparse(s) for s in new_def.body]
    added = [s for s in new_body if s not in old_body]
    removed = [s for s in old_body if s not in new_body]
    if removed:
        parts.append("removed: " + "; ".join(removed))
    if added:
        parts.append("added: " + "; ".join(added))
    old_helpers = [ast.unparse(n) for n in ast.parse(old).body if isinstance(n, ast.FunctionDef) and n.name != "kernel"]
    new_helpers = [ast.unparse(n) for n in ast.parse(new).body if isinstance(n, ast.FunctionDef) and n.name != "kernel"]
    if old_helpers != new_helpers:
        parts.append("helper functions changed")
    return "; ".join(parts) or "formatting of statements"


def dtype_notes(record: dict[str, Any], kernel: generator.GeneratedKernel) -> list[str]:
    notes = []
    cases = [records.evaluate_test_case(record, k) for k in records.test_case_ids(record)]
    for name, ptype in kernel.params:
        per_case = [(c.case_id, c.env[name]) for c in cases if name in c.env]
        last_type = generator.value_type(name, per_case[-1][1])
        if last_type.dtype != ptype.dtype:
            wider = [cid for cid, v in per_case if generator.value_type(name, v).dtype == ptype.dtype]
            notes.append(f"{name} is {ptype.dtype} although the last test case passes {last_type.dtype}: DS-1000 test "
                         f"case {wider[0]} passes a {ptype.dtype.lower()} value")
        if any(isinstance(v, (list, tuple)) for _, v in per_case):
            notes.append(f"DS-1000 passes {name} as a Python list or tuple; the kernel takes an array")
    return notes


def adaptations(record: dict[str, Any], kernel: generator.GeneratedKernel) -> list[str]:
    pid = record["problem_id"]
    items = [f"Body: {kernel.form}."]
    arrays = [f"{n} {t.shape}" for n, t in kernel.params if t.shape is not None]
    if arrays:
        items.append(f"Shapes fixed to DS-1000 test case {kernel.shape_case} ({', '.join(arrays)}); the DS-1000 "
                     "program accepts any shape, a circuit needs one.")
    items += dtype_notes(record, kernel)
    for name, value in kernel.constants.items():
        what = perf.SHAPE_CONSTANTS[pid][name]
        items.append(f"{name} = {generator._constant_literal(value)} is a compile-time constant (DS-1000 test case "
                     f"{kernel.shape_case} value; it sets the {what}). The runtime variant is "
                     f"ds1000_numpy/generated/case{pid:04d}/kernel.py.")
    items += [note + "." for note in kernel.notes]
    if pid in CASE_NOTES:
        items.append(CASE_NOTES[pid])
    return items


# ---------------------------------------------------------------------------
# Output check of the Zinnia mock against NumPy (fixture verification)
# ---------------------------------------------------------------------------


def expected_values(output: Any) -> tuple[list[int], list[float]]:
    ints, floats = [], []
    for x in fixtures.flatten_numbers(output):
        value = np.asarray(x)
        if value.dtype.kind == "c":
            floats += [float(value.real), float(value.imag)]
        elif value.dtype.kind in "iub":
            ints.append(int(value))
        else:
            floats.append(float(value))
    return ints, floats


def public_values(mock: dict[str, Any]) -> tuple[list[int], list[float]]:
    ints, floats = [], []
    for entry in mock["publics"]:
        name, _, value = entry.partition("=")
        (floats.append(int(value) / Q) if name.startswith("public_f_") else ints.append(int(value)))
    for entry in mock["elided"]:
        (ints.append(int(entry["Int"])) if "Int" in entry else floats.append(float(entry["Float"])))
    return ints, floats


def contained(expected: list[float], available: list[float], tol: dict[str, float]) -> list[float]:
    """Multiset containment within tolerance (sorted sweep, nearest unused
    candidate); returns the expected values that found no partner."""
    pool = sorted(available)
    used = [False] * len(pool)
    missing = []
    start = 0
    for e in sorted(expected):
        bound = tol["atol"] + tol["rtol"] * abs(e)
        while start < len(pool) and (used[start] or pool[start] < e - bound):
            start += 1
        best = None
        for j in range(start, len(pool)):
            if pool[j] > e + bound:
                break
            if not used[j] and (best is None or abs(pool[j] - e) < abs(pool[best] - e)):
                best = j
        if best is None:
            missing.append(e)
        else:
            used[best] = True
    return missing


def mock_check(mock: dict[str, Any] | None, output: Any, tol: dict[str, float] | None) -> str | None:
    """None when the mock accepted the witness and its publics contain every NumPy output value."""
    if mock is None:
        return "not run"
    if mock.get("error"):
        return mock["error"]
    if not mock["satisfied"]:
        return "mock prover rejected the honest witness"
    exp_ints, exp_floats = expected_values(output)
    pub_ints, pub_floats = public_values(mock)
    missing_ints = contained([float(v) for v in exp_ints], [float(v) for v in pub_ints] + pub_floats,
                             {"atol": 0.0, "rtol": 0.0})
    missing_floats = contained(exp_floats, pub_floats + [float(v) for v in pub_ints], tol or COPY_TOLERANCE)
    if missing_ints or missing_floats:
        return f"public values do not contain NumPy outputs {(missing_ints + missing_floats)[:6]}"
    return None


def tolerance_for(pid: int, output: Any) -> dict[str, float] | None:
    _, floats = expected_values(output)
    if not floats:
        return None
    return dict(COPY_TOLERANCE if pid in perf.DATA_MOVEMENT_ONLY else ARITHMETIC_TOLERANCE)


# ---------------------------------------------------------------------------
# The case set
# ---------------------------------------------------------------------------


def extension_sha256() -> str:
    return sha256(next((REPO / "zinnia").glob("_zinnia_core*.so")))


def probe_info() -> dict[str, Any]:
    commit = git("rev-parse", "HEAD").strip()
    return {
        "commit": commit,
        "compiler_sources": "unchanged since e58e9a6 (src/, zinnia/ as of the eval-redo snapshot); "
                            f"git diff e58e9a6 -- src zinnia is {'empty' if not git('diff', 'e58e9a6', '--', 'src', 'zinnia').strip() else 'NOT empty'}",
        "extension": "zinnia/_zinnia_core*.so built from those sources (release, halo2-base and noir backends)",
        "extension_sha256": extension_sha256(),
        "date": datetime.date.today().isoformat(),
        "how": "tools/zinnia_probe.py: ZKCircuit.from_method(kernel).compile() in a child process, 300 s limit, "
               "8 GB memory cap; fixtures mock-proved with backend='mock'",
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_entries(write_fixtures: bool, reuse: dict[str, Any] | None) -> list[dict[str, Any]]:
    """`reuse` maps case -> entry of an earlier case set made with the same Zinnia
    extension; its Zinnia results are kept for kernels and fixtures with the same hash."""
    manifest = records.load_manifest()
    in_scope = {perf.problem_id(d): d for d in perf.case_dirs() if manifest[perf.problem_id(d)]["status"] == "in_scope"}
    kernels = {pid: perf.expected_kernel(manifest[pid]) for pid in in_scope}
    duplicates = perf.find_duplicates({pid: k.source for pid, k in kernels.items()
                                       if pid not in perf.DEGENERATE})
    entries = []
    for case_dir in perf.case_dirs():
        pid = perf.problem_id(case_dir)
        record = manifest[pid]
        entry: dict[str, Any] = {"case": case_dir.name, "problem_id": pid}
        print(case_dir.name, flush=True)
        if record["status"] != "in_scope":
            evidence = "; ".join(f"{rule}: {', '.join(items)}" for rule, items in record["scope_rules"].items())
            entry.update(status="out_of_scope", reason=evidence,
                         decision_source="tools/ds1000_scope.py -> ds1000_numpy/manifest.json (rules: datasets/FILTERING.md)",
                         duplicate_of=None, data_movement_only=None, kernel=None, adaptations=[],
                         kernel_change="kernel.py kept for reference with an OUT OF SCOPE header; it is not used",
                         float_tolerance=None, zinnia=None, fixture=None)
            entries.append(entry)
            continue
        kernel = kernels[pid]
        entry["kernel"] = {"params": [{"name": n, "annotation": t.annotation()} for n, t in kernel.params],
                           "constants": {n: generator._plain(np.asarray(v).tolist()) for n, v in kernel.constants.items()},
                           "shape_test_case": kernel.shape_case}
        entry["adaptations"] = adaptations(record, kernel)
        entry["kernel_change"] = kernel_change(previous_kernel(case_dir.name), kernel.source)
        entry["duplicate_of"] = None
        entry["data_movement_only"] = pid in perf.DATA_MOVEMENT_ONLY
        if entry["data_movement_only"]:
            entry["data_movement"] = perf.DATA_MOVEMENT_ONLY[pid]

        entry["kernel_sha256"] = sha256(case_dir / "kernel.py")
        cached = reuse.get(case_dir.name) if reuse is not None else None
        if cached and cached.get("kernel_sha256") == entry["kernel_sha256"] and cached.get("zinnia"):
            probe = {"compiled": cached["zinnia"]["compiles"], "compile_error": cached["zinnia"]["error"]}
        else:
            cached = None
            probe = zinnia_probe.run(case_dir / "kernel.py")
        entry["zinnia"] = {"compiles": probe["compiled"], "error": probe["compile_error"]}

        if pid in perf.DEGENERATE:
            entry.update(status="degenerate", reason=perf.DEGENERATE[pid],
                         decision_source="hand review, tools/ds1000_perf.py DEGENERATE (audit DB-01, DB-08)")
        elif pid in duplicates:
            dup = duplicates[pid]
            entry.update(status="duplicate", reason=dup.reason, duplicate_of=f"ds1000_case{dup.of:04d}",
                         decision_source="tools/ds1000_perf.py find_duplicates" +
                         (" (SEMANTIC_DUPLICATES, hand review)" if pid in perf.SEMANTIC_DUPLICATES else
                          " (equal kernels after renaming)"))
        elif not probe["compiled"]:
            entry.update(status="zinnia_rejects", reason=probe["compile_error"],
                         decision_source="Zinnia compile probe (see zinnia_probe at the top of this file)")
        else:
            entry.update(status="perf", reason="no exclusion rule applies", decision_source="rules above")

        entry["float_tolerance"] = None
        entry["fixture"] = None
        if entry["status"] in ("perf", "zinnia_rejects") and pid in fixtures.DRAWS:
            fixture = fixtures.build_fixture(case_dir, record)
            entry["float_tolerance"] = tolerance_for(pid, fixture.outputs["B"])
            entry["output_kind"] = "float" if entry["float_tolerance"] else "exact"
            fixture_hash = hashlib.sha256(json.dumps(fixture.data, sort_keys=True).encode()).hexdigest()
            if entry["status"] == "perf" and cached and cached.get("fixture_sha256") == fixture_hash \
                    and "mock_vs_numpy" in cached["zinnia"]:
                entry["fixture"] = "fixture.json"
                entry["fixture_sha256"] = fixture_hash
                entry["zinnia"]["mock_vs_numpy"] = cached["zinnia"]["mock_vs_numpy"]
                if write_fixtures:
                    (case_dir / "fixture.json").write_text(fixtures.format_fixture(fixture.data), encoding="utf-8")
            elif entry["status"] == "perf" and cached and cached["status"] == "zinnia_rejects":
                entry.update(status="zinnia_rejects", reason=cached["reason"], decision_source=cached["decision_source"])
            elif entry["status"] == "perf":
                mocks = {name: run_mock(case_dir, fixture, name) for name in fixtures.INPUT_NAMES}
                errors = [m.get("error") for m in mocks.values()]
                if all(errors):
                    entry.update(status="zinnia_rejects", reason=f"compiles, but Zinnia refuses to build a witness "
                                 f"for either fixture input: {errors[0]}",
                                 decision_source="Zinnia mock prove of both fixture inputs (zinnia_probe)")
                else:
                    if write_fixtures:
                        (case_dir / "fixture.json").write_text(fixtures.format_fixture(fixture.data),
                                                               encoding="utf-8")
                    entry["fixture"] = "fixture.json"
                    entry["fixture_sha256"] = fixture_hash
                    entry["zinnia"]["mock_vs_numpy"] = {
                        name: mock_check(mocks[name], fixture.outputs[name], entry["float_tolerance"]) or "ok"
                        for name in fixtures.INPUT_NAMES}
        elif entry["status"] == "perf":
            raise SystemExit(f"{case_dir.name}: perf case without a fixture draw")
        entries.append(entry)
    return entries


def run_mock(case_dir: Path, fixture: fixtures.Fixture, name: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch) / "fixture.json"
        path.write_text(json.dumps(fixture.data), encoding="utf-8")
        result = zinnia_probe.run(case_dir / "kernel.py", path, name)
    return result["mock"] or {"error": result["compile_error"]}


def ordered(entry: dict[str, Any]) -> dict[str, Any]:
    keys = ["case", "problem_id", "status", "reason", "decision_source", "duplicate_of", "data_movement_only",
            "data_movement", "kernel", "adaptations", "kernel_change", "output_kind", "float_tolerance", "fixture",
            "zinnia", "kernel_sha256", "fixture_sha256"]
    return {k: entry[k] for k in keys if k in entry}


def write_case_set(entries: list[dict[str, Any]]) -> None:
    counts = Counter(e["status"] for e in entries)
    document = {
        "case_set": "ds1000_perf",
        "version": 1,
        "description": "The DS-1000 cross-system performance set: every datasets/ds1000/ds1000_case<NNNN> folder "
                       "with one status. Written by tools/perf_case_set.py; edit the decisions in tools/ds1000_perf.py "
                       "and tools/perf_fixtures.py, not this file.",
        "rules": RULES,
        "zinnia_probe": probe_info(),
        "counts": {status: counts[status] for status in STATUS_ORDER},
        "data_movement_only_perf_cases": sum(1 for e in entries if e["status"] == "perf" and e.get("data_movement_only")),
        "zinnia_output_mismatches": {
            "meaning": "perf cases where Zinnia's mock prover accepts the honest witness but its public outputs do "
                       "not match NumPy on a fixture input: Zinnia computes a different function (compiler bug)",
            "cases": sorted(e["case"] for e in entries
                            if any(v != "ok" for v in (e.get("zinnia") or {}).get("mock_vs_numpy", {}).values())),
        },
        "cases": [ordered(e) for e in entries],
    }
    CASE_SET.parent.mkdir(exist_ok=True)
    CASE_SET.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")


def check_kernels() -> int:
    manifest = records.load_manifest()
    bad = 0
    for case_dir in perf.case_dirs():
        pid = perf.problem_id(case_dir)
        if manifest[pid]["status"] != "in_scope":
            continue
        expected = perf.expected_kernel(manifest[pid]).source
        actual = (case_dir / "kernel.py").read_text(encoding="utf-8")
        if actual != expected:
            bad += 1
            print(f"{case_dir.name}: kernel.py differs from the generator output "
                  f"(generate_ds1000_kernels.generate(record, {list(perf.SHAPE_CONSTANTS.get(pid, {}))}))")
    print(f"{bad} kernel(s) differ")
    return bad


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="compare kernels with the generator output")
    parser.add_argument("--write", action="store_true", help="probe Zinnia and write the case set and fixtures")
    parser.add_argument("--reuse-zinnia", action="store_true",
                        help="keep Zinnia results of the existing case set for unchanged kernels and fixtures "
                             "(only when the Zinnia extension is the same binary)")
    args = parser.parse_args()
    if args.check:
        sys.exit(1 if check_kernels() else 0)
    if args.write:
        reuse = None
        if args.reuse_zinnia and CASE_SET.exists():
            previous = json.loads(CASE_SET.read_text())
            if previous["zinnia_probe"].get("extension_sha256") == extension_sha256():
                reuse = {e["case"]: e for e in previous["cases"]}
        entries = build_entries(write_fixtures=True, reuse=reuse)
        write_case_set(entries)
        print(json.dumps(Counter(e["status"] for e in entries), indent=1))


if __name__ == "__main__":
    main()
