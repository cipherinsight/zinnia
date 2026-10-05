#!/usr/bin/env python3
"""Decide which DS-1000 NumPy programs are in scope, using `scope_rules.py`.

A DS-1000 program is the reference solution as DS-1000 runs it: inside the
record's exec_context, fed by its test inputs. So the rules are applied to

* the solution text (with the `def f(...)` wrapper DS-1000 adds, if any) and
  the exec_context suffix;
* the prelude statements that compute an input the solution reads (problem
  0477 reads singular values its prelude computes with `np.linalg.svd`);
* the prelude imports whose names the solution reads (`scipy`, `pd`, ...);
* every DS-1000 test input: an input that is a pandas, torch or tensorflow
  object counts as using that library (rule `imports`); a string, object,
  ragged or datetime input fires `non_numeric_data`; a NaN or infinity fires
  `non_finite_values`.

No rule asks whether Zinnia compiles a program. This script writes
`ds1000_numpy/manifest.json` (all 220 programs with status and rule evidence),
the per-program `record.json`/`reference.py`, and the inventory
`ds1000_numpy/SCOPE.md`.
"""

from __future__ import annotations

import argparse
import ast
import gzip
import json
import sys
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ds1000_records as records  # noqa: E402
import generate_ds1000_kernels as generator  # noqa: E402
from scope_rules import RULES, RuleHits, text_rules  # noqa: E402

DATASET_URL = "https://raw.githubusercontent.com/xlang-ai/DS-1000/main/data/ds1000.jsonl.gz"
OUT_DIR = records.DATASETS / "ds1000_numpy"
CACHE = OUT_DIR / "ds1000.jsonl.gz"
LIBRARY_ROOTS = {"pandas": "pandas", "torch": "torch", "tf": "tensorflow", "tensorflow": "tensorflow",
                 "scipy": "scipy", "sklearn": "sklearn"}


@dataclass
class ScopeResult:
    status: str                                     # in_scope | out_of_scope
    rules: dict[str, list[str]]                      # rule -> evidence
    notes: list[str] = field(default_factory=list)


def _import_module_of(name: str, prelude: list[ast.stmt]) -> str:
    for node in prelude:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if (alias.asname or alias.name.split(".")[0]) == name:
                    return alias.name
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if (alias.asname or alias.name) == name:
                    return f"{node.module}.{alias.name}"
    return name


def scope_record(record: dict[str, Any]) -> ScopeResult:
    hits = RuleHits()
    notes: list[str] = []
    context = records.parse_exec_context(record["code_context"])
    hits.merge(text_rules(ast.parse(records.program_source(record, context))))

    try:
        inputs = records.program_inputs(record)
    except records.RecordError as exc:
        notes.append(f"inputs not derivable: {exc}")
        return ScopeResult("out_of_scope" if hits.rules else "in_scope", hits.sorted(), notes)

    for name in inputs.imported_reads:
        module = _import_module_of(name, context.prelude)
        if module.split(".")[0] not in {"numpy", "math"}:
            hits.add("imports", f"solution uses `{name}` from the exec_context import of {module}")
    for node in context.prelude:
        binds_input = set(records.assigned_names(node)) & set(inputs.names)
        if binds_input and not records.reads_test_input(node):
            prelude_hits = text_rules(node)
            hits.merge(prelude_hits)
            notes.append(f"input {', '.join(sorted(binds_input))} is computed by the exec_context: "
                         f"{ast.unparse(node)}")

    try:
        _, last_case = generator.infer_types(record, inputs.names)
        if last_case.pandas_used:
            notes.append("a test input is a pandas Series of numbers; the solution only indexes it "
                         "like a NumPy array")
    except generator.UntypeableInput as exc:
        rule = {"non_numeric": "non_numeric_data", "non_finite": "non_finite_values"}.get(exc.category)
        if rule:
            hits.add(rule, f"test input {exc.name}: {exc.detail}")
        else:
            notes.append(f"test input {exc.name}: {exc.detail}")
    except records.RecordError as exc:
        message = str(exc)
        library = next((lib for key, lib in LIBRARY_ROOTS.items() if key in message), None)
        if library:
            hits.add("imports", f"a test input is a {library} object ({message})")
        else:
            notes.append(f"test inputs not evaluable: {message}")
    return ScopeResult("out_of_scope" if hits.rules else "in_scope", hits.sorted(), notes)


def load_records(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def row_for(record: dict[str, Any]) -> dict[str, Any]:
    meta = record["metadata"]
    row = {
        "problem_id": meta["problem_id"],
        "library_problem_id": meta["library_problem_id"],
        "library": meta["library"],
        "reference_code": record["reference_code"],
        "prompt": record["prompt"],
        "code_context": record["code_context"],
        "test_case_cnt": meta.get("test_case_cnt"),
        "perturbation_type": meta.get("perturbation_type"),
        "perturbation_origin_id": meta.get("perturbation_origin_id"),
    }
    scope = scope_record(row)
    row.update({"status": scope.status, "scope_rules": scope.rules, "scope_notes": scope.notes})
    return row


def write_inventory(path: Path, rows: list[dict[str, Any]]) -> None:
    counts = Counter(row["status"] for row in rows)
    rule_counts = Counter(rule for row in rows for rule in row["scope_rules"])
    lines = [
        "# DS-1000 NumPy scope inventory",
        "",
        "Written by `benchmarking/datasets/tools/ds1000_scope.py`; the rules are stated in",
        "`benchmarking/datasets/FILTERING.md`. Do not edit by hand.",
        "",
        f"- Source: `{DATASET_URL}`",
        f"- NumPy programs: {len(rows)}",
        f"- In scope: {counts['in_scope']}",
        f"- Out of scope: {counts['out_of_scope']}",
        "",
        "| Rule | Programs it fires on |",
        "| --- | ---: |",
    ]
    lines += [f"| `{rule}` | {rule_counts[rule]} |" for rule in RULES if rule_counts[rule]]
    lines += ["", "## Out of scope", "", "| Problem | Rules and evidence |", "| --- | --- |"]
    for row in rows:
        if row["status"] == "out_of_scope":
            evidence = "; ".join(f"`{rule}`: {', '.join(items)}" for rule, items in row["scope_rules"].items())
            lines.append(f"| {row['problem_id']:04d} | {evidence.replace('|', '/')} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset-cache", type=Path, default=CACHE)
    parser.add_argument("--download", action="store_true", help=f"fetch {DATASET_URL} if the cache is missing")
    args = parser.parse_args()

    if not args.dataset_cache.exists():
        if not args.download:
            raise SystemExit(f"{args.dataset_cache} missing; pass --download")
        args.dataset_cache.write_bytes(urllib.request.urlopen(DATASET_URL, timeout=60).read())
    numpy_records = [r for r in load_records(args.dataset_cache) if r["metadata"]["library"] == "Numpy"]
    rows = sorted((row_for(r) for r in numpy_records), key=lambda row: row["problem_id"])

    (OUT_DIR / "manifest.json").write_text(json.dumps({"programs": rows}, indent=2, sort_keys=True) + "\n",
                                           encoding="utf-8")
    for row in rows:
        case_dir = OUT_DIR / f"case{row['problem_id']:04d}"
        case_dir.mkdir(exist_ok=True)
        (case_dir / "reference.py").write_text(row["reference_code"].rstrip() + "\n", encoding="utf-8")
        (case_dir / "record.json").write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_inventory(OUT_DIR / "SCOPE.md", rows)
    counts = Counter(row["status"] for row in rows)
    print(json.dumps({"programs": len(rows), **counts}, indent=2))


if __name__ == "__main__":
    main()
