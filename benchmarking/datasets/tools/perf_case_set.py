#!/usr/bin/env python3
"""Write `datasets/case_sets/ds1000_perf.json`: one entry per performance-set case
with its status, the reason and the source of the decision, its adaptations,
the output tolerance, and what Zinnia did with it.

    perf_case_set.py --check          # kernels equal the generator's output?
    perf_case_set.py --write          # apply frozen RQ1 decisions; NumPy fixture checks only

`--write` refreshes the inventory from the frozen RQ1_ACCEPTED decision in
`ds1000_perf.py`, retaining historical compiler observations without running
Zinnia. It verifies the generated fixtures with NumPy. By default it writes
only the case set; `--write-fixtures` also materializes the generated fixtures.
Historical statuses and duplicate links are retained in `former_*` fields.
No compilation, witness generation, key generation or proof is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ds1000_perf as perf  # noqa: E402
import ds1000_records as records  # noqa: E402
import generate_ds1000_kernels as generator  # noqa: E402
import perf_fixtures as fixtures  # noqa: E402

CASE_SET = records.DATASETS / "case_sets" / "ds1000_perf.json"

STATUS_ORDER = ["out_of_scope", "degenerate", "duplicate", "zinnia_rejects", "perf"]

COPY_TOLERANCE = {"atol": 2.0 ** -32, "rtol": 0.0}
ARITHMETIC_TOLERANCE = {"atol": 1e-6, "rtol": 1e-6}

RULES = {
    "status_order": "RQ1_ACCEPTED in tools/ds1000_perf.py takes precedence: every accepted program is perf. "
                    "Other programs retain their historical status (including the eight legacy perf cases "
                    "rejected in RQ1). Former exclusion rules below are historical, not new filters.",
    "out_of_scope": "The DS-1000 program is out of scope in ds1000_numpy/manifest.json: it imports or uses a module "
                    "other than numpy (incl. submodules) and math (the author's rule), or another rule of "
                    "datasets/FILTERING.md fires (non-numeric data, NaN/infinity, ...), checked on the solution text "
                    "and on every DS-1000 test input.",
    "degenerate": "No output depends on a private input: every output is fixed by the parameter types and the "
                  "constants, so there is nothing to prove.",
    "duplicate": "The kernel computes the same function as a case with a lower id; the lower id is kept. Three "
                 "tests: the kernels are identical after renaming local variables (same types and constants); or, "
                 "fed the same input values, the outputs are a fixed reordering or reshaping of the other case's "
                 "(numeric test on both cases' fixtures); or both only copy input cells to outputs (no arithmetic "
                 "or comparison) and copy the same number of cells the same number of times with the same constant "
                 "outputs, so they are the same circuit up to which cells are read (input cells no output reads are "
                 "ignored: no system needs to load them).",
    "zinnia_rejects": "Zinnia at the commit in `zinnia_probe` refuses the kernel: compilation fails, or it compiles "
                      "but Zinnia refuses to build a witness for either fixture input, or neither the halo2-base nor "
                      "the Noir backend synthesizes the circuit (error recorded). Provisional: compiler fixes on other "
                      "branches may change it. A case only one backend refuses stays perf and is listed under "
                      "zinnia_backend_gaps.",
    "perf": "Every program in tools/ds1000_perf.py RQ1_ACCEPTED, plus the eight retained legacy perf cases. "
            "Duplicates and constant-output programs are included; former_status and former_duplicate_of "
            "preserve the original classification. Analysis filters legacy perf programs rejected in RQ1.",
    "kernels": "Every in-scope kernel is the output of tools/generate_ds1000_kernels.py (shapes from the record's "
               "last DS-1000 test case; each dtype the narrowest of Boolean < Integer < Float < Complex holding the "
               "parameter in every test case; returns the value DS-1000 checks) with the shape parameters listed "
               "in tools/ds1000_perf.py bound as compile-time constants and the existing 0318/0319 permutation "
               "preconditions documented in FILTERING.md. tools/perf_case_set.py --check verifies it.",
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
        items.append(f"{name} = {generator.constant_literal(value)} is a compile-time constant (DS-1000 test case "
                     f"{kernel.shape_case} value; it sets the {what}). The runtime variant is "
                     f"ds1000_numpy/generated/case{pid:04d}/kernel.py.")
    items += [note + "." for note in kernel.notes]
    if pid in CASE_NOTES:
        items.append(CASE_NOTES[pid])
    return items


# ---------------------------------------------------------------------------
# Output kinds and NumPy fixture tolerance
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


def tolerance_for(pid: int, output: Any) -> dict[str, float] | None:
    _, floats = expected_values(output)
    if not floats:
        return None
    return dict(COPY_TOLERANCE if pid in perf.DATA_MOVEMENT_ONLY else ARITHMETIC_TOLERANCE)


# ---------------------------------------------------------------------------
# The case set
# ---------------------------------------------------------------------------


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_entries(write_fixtures: bool, reuse: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Apply the author's frozen RQ1 decision without invoking the compiler.

    `reuse` is the historical inventory, not a cache authorizing new probes.
    Existing non-accepted entries keep their status; in particular the eight
    legacy perf entries rejected in RQ1 are not removed. Regeneration is
    idempotent and never overwrites the first recorded former classification.
    """
    if reuse is None:
        if not CASE_SET.exists():
            raise SystemExit("the historical case set is required to preserve former decisions")
        reuse = {e["case"]: e for e in json.loads(CASE_SET.read_text())["cases"]}
    manifest = records.load_manifest()
    folders = {perf.problem_id(d): d for d in perf.case_dirs()}
    missing = perf.RQ1_ACCEPTED - folders.keys()
    if missing:
        raise SystemExit(f"RQ1-accepted programs without performance folders: {sorted(missing)}")
    missing_folders = set(reuse) - {d.name for d in folders.values()}
    if missing_folders:
        raise SystemExit(f"historical cases without performance folders: {sorted(missing_folders)}")
    entries = []
    for pid, case_dir in sorted(folders.items()):
        previous = reuse.get(case_dir.name)
        if previous is None and pid not in perf.RQ1_ACCEPTED:
            raise SystemExit(f"{case_dir.name}: needs an explicit selection decision")
        # Deep-copy because compiler observations are historical evidence.
        entry = json.loads(json.dumps(previous)) if previous else {
            "case": case_dir.name, "problem_id": pid, "status": "absent",
            "reason": "no performance folder after excluded_numpy_family was removed (audit DB-09)",
            "decision_source": "datasets/FILTERING.md: rules removed on 2026-10-06",
            "duplicate_of": None,
            "zinnia": {"compiles": None, "error": None,
                       "note": "not probed on this performance variant; selection comes from frozen RQ1 acceptance"},
        }
        entry.setdefault("former_status", entry["status"])
        entry.setdefault("former_duplicate_of", entry.get("duplicate_of"))
        entry.setdefault("former_reason", entry.get("reason"))
        entry.setdefault("former_decision_source", entry.get("decision_source"))
        if pid in perf.RQ1_ACCEPTED:
            if manifest[pid]["status"] != "in_scope":
                raise SystemExit(f"{case_dir.name}: frozen RQ1 acceptance conflicts with current scope")
            kernel = perf.expected_kernel(manifest[pid])
            if (case_dir / "kernel.py").read_text() != kernel.source:
                raise SystemExit(f"{case_dir.name}: kernel differs from generated performance variant")
            if previous and previous.get("kernel_sha256") != sha256(case_dir / "kernel.py"):
                raise SystemExit(f"{case_dir.name}: kernel changed; historical compiler evidence cannot be reused")
            entry.update(status="perf", duplicate_of=None,
                         reason="accepted in the frozen RQ1 program list; include duplicates and constant outputs",
                         decision_source="author decision 2026-10-09, tools/ds1000_perf.py RQ1_ACCEPTED",
                         data_movement_only=pid in perf.DATA_MOVEMENT_ONLY)
            if pid in perf.DATA_MOVEMENT_ONLY:
                entry["data_movement"] = perf.DATA_MOVEMENT_ONLY[pid]
            else:
                entry.pop("data_movement", None)
            if not previous:
                entry["kernel"] = {
                    "params": [{"name": n, "annotation": t.annotation()} for n, t in kernel.params],
                    "constants": {n: generator.plain_value(np.asarray(v).tolist())
                                  for n, v in kernel.constants.items()},
                    "shape_test_case": kernel.shape_case,
                }
                entry["adaptations"] = adaptations(manifest[pid], kernel)
                entry["kernel_change"] = "new performance folder, mechanically generated from DS-1000"
                entry["kernel_sha256"] = sha256(case_dir / "kernel.py")
            fixture = fixtures.build_fixture(case_dir, manifest[pid])
            fixture_hash = hashlib.sha256(json.dumps(fixture.data, sort_keys=True).encode()).hexdigest()
            # Keep old probe observations only for the very same fixture. A
            # selection change must not silently relabel stale witness evidence.
            if entry.get("fixture_sha256") != fixture_hash:
                for key in ("mock_vs_numpy", "backends"):
                    if key in entry["zinnia"]:
                        entry.setdefault("former_zinnia_fixture_evidence", {})[key] = entry["zinnia"].pop(key)
            entry.update(fixture="fixture.json", fixture_sha256=fixture_hash,
                         float_tolerance=tolerance_for(pid, fixture.outputs["B"]))
            entry["output_kind"] = "float" if entry["float_tolerance"] else "exact"
            if write_fixtures:
                (case_dir / "fixture.json").write_text(fixtures.format_fixture(fixture.data), encoding="utf-8")
        entries.append(entry)
    return entries


def ordered(entry: dict[str, Any]) -> dict[str, Any]:
    keys = ["case", "problem_id", "status", "reason", "decision_source", "duplicate_of",
            "former_status", "former_duplicate_of", "former_reason", "former_decision_source",
            "former_zinnia_fixture_evidence", "data_movement_only",
            "data_movement", "kernel", "adaptations", "kernel_change", "output_kind", "float_tolerance", "fixture",
            "zinnia", "kernel_sha256", "fixture_sha256"]
    return {k: entry[k] for k in keys if k in entry}


def write_case_set(entries: list[dict[str, Any]]) -> None:
    counts = Counter(e["status"] for e in entries)
    previous = json.loads(CASE_SET.read_text())
    document = {
        "case_set": "ds1000_perf",
        "version": 2,
        "description": "The DS-1000 cross-system performance set: every datasets/ds1000/ds1000_case<NNNN> folder "
                       "with one status. Written by tools/perf_case_set.py; edit the decisions in tools/ds1000_perf.py "
                       "and tools/perf_fixtures.py, not this file.",
        "rules": RULES,
        "selection": {
            "source": "tools/ds1000_perf.py RQ1_ACCEPTED (frozen from eval-redo/all-work/rq1-ds-accepted.txt)",
            "accepted_programs": len(perf.RQ1_ACCEPTED),
            "how": "author selection update; NumPy fixture validation only; no new Zinnia probes",
            "historical_evidence": "zinnia_probe and per-case zinnia fields are retained historical observations, "
                                   "not validation of the present compiler or the new selection",
        },
        "zinnia_probe": previous.get("zinnia_probe"),
        "counts": {status: counts[status] for status in STATUS_ORDER},
        "data_movement_only_perf_cases": sum(1 for e in entries if e["status"] == "perf" and e.get("data_movement_only")),
        "zinnia_backend_gaps": {
            "meaning": "perf cases whose circuit one Zinnia backend cannot synthesize (emit_structural on input B)",
            "cases": {e["case"]: {b: v for b, v in e["zinnia"].get("backends", {}).items() if not v.startswith("ok")}
                      for e in entries if e["status"] == "perf"
                      and any(not v.startswith("ok") for v in e["zinnia"].get("backends", {}).values())},
        },
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
    parser.add_argument("--write", action="store_true", help="apply frozen RQ1 decisions and write the case set (no compiler)")
    parser.add_argument("--write-fixtures", action="store_true",
                        help="with --write, also materialize accepted cases' fixtures; default writes only the case set")
    parser.add_argument("--reuse-zinnia", action="store_true",
                        help="compatibility flag: historical compiler observations are always retained without new probes")
    args = parser.parse_args()
    if args.check:
        sys.exit(1 if check_kernels() else 0)
    if args.write_fixtures and not args.write:
        parser.error("--write-fixtures requires --write")
    if args.write:
        entries = build_entries(write_fixtures=args.write_fixtures, reuse=None)
        write_case_set(entries)
        print(json.dumps(Counter(e["status"] for e in entries), indent=1))


if __name__ == "__main__":
    main()
