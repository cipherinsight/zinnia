"""The zkVM programs of the scaled entries may differ from the originals only in size constants.

Every scaled folder `DATA/cap<C>/<program>/` holds a `zkvm.rs`, which the SP1
and RISC Zero guests of `<program>__<function>` are built from. The check
diffs it against `datasets/{npbench,pythran}/<program>/zkvm.rs`:

* identical: the program reads every size from its input (the NPBench/Pythran
  ports are shape-generic: arrays carry their shapes, and the guest commits
  the input shapes, so the proof binds them); the original guest binary runs
  the scaled point as it is;
* otherwise every changed line must be a `const NAME: TYPE = <integer>;`
  definition (a size, or a constant computed from sizes by a formula); any
  other change makes the point "baseline not scalable".

The verdict per (case, cap) goes to the output JSON and to stdout.
"""

from __future__ import annotations

import difflib
import json
import re
from pathlib import Path

from benchmarking.scaling_npb.generate import BENCHMARKING

SIZE_CONSTANT = re.compile(r"^\s*(pub\s+)?const\s+[A-Z][A-Z0-9_]*\s*:\s*\w+\s*=\s*[0-9_]+\s*;\s*(//.*)?$")


def compare(original: str, scaled: str) -> tuple[str, list[str]]:
    """("identical" | "size constants" | "not scalable", the changed lines)."""
    if original == scaled:
        return "identical", []
    changed = [line for line in difflib.unified_diff(original.splitlines(), scaled.splitlines(), lineterm="", n=0)
               if line[:1] in "+-" and not line.startswith(("+++", "---"))]
    if all(SIZE_CONSTANT.match(line[1:]) for line in changed):
        return "size constants", changed
    return "not scalable", changed


def main(data: Path, out: Path | None) -> int:
    records = []
    for case_set in sorted((data / "case_sets").glob("npbench_pythran_cap*.json")):
        for case in json.loads(case_set.read_text())["cases"]:
            if case["status"] != "perf":
                continue
            scaled = data / case["data_folder"] / "zkvm.rs"
            original = BENCHMARKING / "datasets" / case["group"] / case["program"] / "zkvm.rs"
            verdict, changed = compare(original.read_text(), scaled.read_text())
            records.append({"case": case["case"], "cap": case["scaling"]["cap"], "verdict": verdict,
                            "changed_lines": changed})
            print(f"{case['case']:<72} {verdict}")
    counts: dict[str, int] = {}
    for record in records:
        counts[record["verdict"]] = counts.get(record["verdict"], 0) + 1
    print(json.dumps(counts))
    if out:
        out.write_text(json.dumps({"counts": counts, "records": records}, indent=1) + "\n")
    return 0 if "not scalable" not in counts else 1
