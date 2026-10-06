#!/usr/bin/env python3
"""Write `datasets/case_sets/npbench_pythran_perf.json`: the NPBench and Pythran
entry functions for the performance comparison of Zinnia with the zkVMs.

    .venv/bin/python benchmarking/datasets/tools/npbench_pythran_perf.py \\
        RESULTS.json [RESULTS.json ...] [--check]

RESULTS.json is the `results.json` of an acceptance run (`python -m
benchmarking.harness accept`), or any file with the same `results` rows
(`program`, `function`, `outcome`, `construct`, `detail`). With several files a
later row for the same entry function replaces an earlier one. `--check` only
reports whether the written case set is what the inputs give.

One case per in-scope NPBench/Pythran entry function of
`case_sets/acceptance.json`. A case is `perf` (measured) only when:

1. the acceptance run accepted it with outputs equal to NumPy's
   (`accepted_correct`); otherwise `zinnia_rejects` (or
   `not_in_acceptance_results`);
2. its program's `zkvm.rs` has a module for the function, registered in the
   zkVM crates (`baselines/zkvm_common/sync_cases.py`); otherwise
   `no_zkvm_port`;
3. the port is not excluded by its audit (`port_excluded`), and the audit
   covers the current `zkvm.rs` (same SHA-256); otherwise `port_not_audited`.
   The audit record is `npbench_pythran_port_audit.json` next to this script;
   see `eval-redo/fix/eval-npbench.md` for the checklist and the findings.

For a `perf` function that returns `None`, the outputs are the array arguments
it writes (`mutated_outputs`, from the port's `// outputs: mutated` line); the
script runs the kernel as NumPy on fixture inputs A and B and fails if the call
changes an array argument the list does not name, or if a function the port
says returns `None` returns a value (or the reverse). `data_movement_only`
cases are listed in DATA_MOVEMENT_ONLY and checked like the DS-1000 ones: every
output value on A and B is an input value or 0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from benchmarking.harness.fixtures import PROVE_INPUT, SETUP_INPUT, load_fixture  # noqa: E402
from benchmarking.harness.outputs import call_outputs  # noqa: E402
from benchmarking.harness.reference import run_kernel_with_arguments  # noqa: E402

BENCHMARKING = REPO / "benchmarking"
DATASETS = BENCHMARKING / "datasets"
ACCEPTANCE_SET = DATASETS / "case_sets" / "acceptance.json"
CASE_SET = DATASETS / "case_sets" / "npbench_pythran_perf.json"
PORT_AUDIT = Path(__file__).with_name("npbench_pythran_port_audit.json")
ZKVM_CASES = BENCHMARKING / "baselines" / "zkvm_common" / "kernels" / "src" / "cases.rs"
DATASETS_IN_SET = ("npbench", "pythran")
ACCEPTED = "accepted_correct"

#: Entry functions whose outputs depend on private inputs only by copying them
#: (no arithmetic or comparison on private values), as `data_movement_only` in
#: ds1000_perf.json. They stay `perf` and are reported apart.
DATA_MOVEMENT_ONLY = {
    "pythran_create_grid/create_grid": "copies x into two columns of a grid of zeros",
    "pythran_multi_export/a": "returns its argument",
    "pythran_periodic_bc/periodic_bc": "copies one plane of f onto another",
    "pythran_perm/permutations": "lists the permutations of the input (fixed positions)",
}

RULES = {
    "status_order": "A case gets the first status whose rule applies, in this order: "
                    "not_in_acceptance_results, zinnia_rejects, no_zkvm_port, port_excluded, "
                    "port_not_audited, perf.",
    "not_in_acceptance_results": "The acceptance results given to the generator have no row for "
                                 "the entry function.",
    "zinnia_rejects": "The acceptance run did not accept the entry function with outputs equal "
                      "to NumPy's (its outcome and construct are in the reason).",
    "no_zkvm_port": "The program's zkvm.rs has no module for the function, or the zkVM crates "
                    "do not register it (run baselines/zkvm_common/sync_cases.py).",
    "port_excluded": "The port audit excludes the port (reason in the audit record).",
    "port_not_audited": "The port audit does not cover the current zkvm.rs (new entry, or the "
                        "file changed since the audit): review it and update the audit record.",
    "perf": "Measured: Zinnia accepts the function, and its SP1/RISC Zero port is audited.",
    "sizes": "Shapes and scalar values are the acceptance fixture's (acceptance.json `sizing`: "
             "every dimension and size scalar at most 16); keys come from input A, proofs from "
             "input B.",
    "mutated_outputs": "A function that returns None outputs the array arguments it writes, in "
                       "parameter order (the port's `// outputs: mutated` line, checked against "
                       "NumPy on inputs A and B).",
    "data_movement_only": "true when the outputs depend on private inputs only by copying them; "
                          "such cases stay perf and are reported apart.",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def case_id(program: str, function: str) -> str:
    """`<program>__<function>`, the name of the zkVM guest binary too."""
    return f"{program}__{function}"


def load_results(paths: list[Path]) -> dict[tuple[str, str], dict]:
    rows: dict[tuple[str, str], dict] = {}
    for path in paths:
        for row in json.loads(path.read_text())["results"]:
            rows[(row["program"], row["function"])] = row
    return rows


def port_module(zkvm_rs: Path, function: str) -> str | None:
    """The text of `pub mod <function> { ... }` in a zkvm.rs, or None."""
    if not zkvm_rs.exists():
        return None
    text = zkvm_rs.read_text()
    match = re.search(rf"^pub mod {re.escape(function)}\s*\{{", text, re.M)
    if match is None:
        return None
    following = re.search(r"^pub mod \w+\s*\{", text[match.end():], re.M)
    return text[match.start():match.end() + following.start() if following else len(text)]


def mutated_outputs(module: str) -> list[str]:
    line = re.search(r"^\s*//\s*outputs:\s*mutated\s+(.+?)\s*$", module, re.M)
    return [name.strip() for name in line.group(1).split(",")] if line else []


def registered_ports() -> set[str]:
    return set(re.findall(r'"(\w+/\w+)"\s*=>', ZKVM_CASES.read_text()))


def check_outputs_contract(kernel: Path, function: str, fixture: Path,
                           written: list[str]) -> list[str]:
    """Problems with the declared outputs on inputs A and B (empty: none)."""
    problems = []
    loaded = load_fixture(fixture)
    for name in (SETUP_INPUT, PROVE_INPUT):
        arguments = loaded.arguments(name)
        result, after = run_kernel_with_arguments(kernel, function, arguments)
        if (result is None) != bool(written):
            problems.append(f"input {name}: returns {'None' if result is None else 'a value'} "
                            f"but the port declares mutated outputs {written}")
            continue
        try:
            call_outputs(result, after, arguments, written)
        except ValueError as exc:
            problems.append(f"input {name}: {exc}")
    return problems


def copy_only_problem(kernel: Path, function: str, fixture: Path, written: list[str]) -> str | None:
    """Why the outputs are not copies of input values (or 0), or None."""
    loaded = load_fixture(fixture)
    for name in (SETUP_INPUT, PROVE_INPUT):
        arguments = loaded.arguments(name)
        result, after = run_kernel_with_arguments(kernel, function, arguments)
        outputs = result if result is not None else [after[n] for n in written]
        pool = {complex(x) for a in arguments for x in np.asarray(a).ravel()} | {0j}
        flat = np.concatenate([np.asarray(o, dtype=complex).ravel()
                               for o in (outputs if isinstance(outputs, (list, tuple))
                                         else [outputs])])
        stray = [x for x in flat if complex(x) not in pool]
        if stray:
            return f"input {name}: output value {stray[0]} is neither an input value nor 0"
    return None


def build_case(entry: dict, result: dict | None, audit: dict, registered: set[str]) -> dict:
    program, function, dataset = entry["program"], entry["function"], entry["dataset"]
    key = f"{program}/{function}"
    folder = DATASETS / dataset / program
    zkvm_rs = folder / "zkvm.rs"
    module = port_module(zkvm_rs, function)
    case = {"case": case_id(program, function), "status": None, "reason": None,
            "group": dataset, "program": program, "function": function,
            "path": str(folder.relative_to(BENCHMARKING)),
            "fixture": Path(entry["fixture"]).name, "shape_generic": True,
            "mutated_outputs": mutated_outputs(module) if module else [],
            "data_movement_only": key in DATA_MOVEMENT_ONLY,
            "float_tolerance": entry["float_tolerance"],
            "acceptance": ({"outcome": result["outcome"], "construct": result.get("construct")}
                           if result else None),
            "zkvm_port": ({"file": str(zkvm_rs.relative_to(BENCHMARKING)),
                           "sha256": sha256(zkvm_rs)} if module else None)}
    excluded = audit.get("excluded", {})
    audited = audit.get("audited", {})
    if result is None:
        case.update(status="not_in_acceptance_results",
                    reason="no row for this entry function in the acceptance results")
    elif result["outcome"] != ACCEPTED:
        construct = f" ({result['construct']})" if result.get("construct") else ""
        case.update(status="zinnia_rejects",
                    reason=f"{result['outcome']}{construct}: {result.get('detail', '')[:200]}")
    elif module is None or key not in registered:
        case.update(status="no_zkvm_port",
                    reason="no module in zkvm.rs" if module is None
                    else "module not registered in the zkVM crates (run sync_cases.py)")
    elif key in excluded:
        case.update(status="port_excluded", reason=excluded[key])
    elif key not in audited or audited[key]["zkvm_rs_sha256"] != case["zkvm_port"]["sha256"]:
        case.update(status="port_not_audited",
                    reason="not in the port audit" if key not in audited
                    else "zkvm.rs changed since the audit")
    else:
        case.update(status="perf", reason=None)
    if case["status"] == "perf":
        kernel = DATASETS / entry["kernel"]
        fixture = DATASETS / entry["fixture"]
        problems = check_outputs_contract(kernel, function, fixture, case["mutated_outputs"])
        if problems:
            raise SystemExit(f"{key}: the port's outputs disagree with NumPy: {problems}")
        copy_problem = copy_only_problem(kernel, function, fixture, case["mutated_outputs"])
        if case["data_movement_only"] and copy_problem:
            raise SystemExit(f"{key}: listed as data movement, but {copy_problem}")
        if not case["data_movement_only"] and copy_problem is None:
            print(f"note: {key}: every output is an input value or 0 on A and B; "
                  "data movement only?", file=sys.stderr)
        case["data_movement"] = DATA_MOVEMENT_ONLY.get(key)
    return case


def build_case_set(result_paths: list[Path]) -> dict:
    acceptance = json.loads(ACCEPTANCE_SET.read_text())
    results = load_results(result_paths)
    audit = json.loads(PORT_AUDIT.read_text())
    registered = registered_ports()
    entries = [e for e in acceptance["entries"]
               if e["dataset"] in DATASETS_IN_SET and e.get("status") == "in_scope"
               and e.get("entry_function", True)]
    cases = sorted((build_case(e, results.get((e["program"], e["function"])), audit, registered)
                    for e in entries), key=lambda c: c["case"])
    counts: dict[str, dict[str, int]] = {}
    for case in cases:
        by_status = counts.setdefault(case["group"], {})
        by_status[case["status"]] = by_status.get(case["status"], 0) + 1
    return {
        "case_set": "npbench_pythran_perf",
        "version": 1,
        "description": "The NPBench and Pythran entry functions Zinnia accepts that have an "
                       "audited SP1/RISC Zero port: the zkVM performance comparison beyond "
                       "DS-1000. Written by datasets/tools/npbench_pythran_perf.py; do not edit.",
        "rules": RULES,
        "sources": {
            "acceptance_results": [{"path": str(p), "sha256": sha256(p)} for p in result_paths],
            "acceptance_case_set": {"path": str(ACCEPTANCE_SET.relative_to(BENCHMARKING)),
                                    "sha256": sha256(ACCEPTANCE_SET)},
            "port_audit": {"path": str(PORT_AUDIT.relative_to(BENCHMARKING)),
                           "sha256": sha256(PORT_AUDIT)},
        },
        "counts": counts,
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("results", nargs="+", type=Path, help="acceptance results.json file(s)")
    parser.add_argument("--check", action="store_true",
                        help="compare with the written case set instead of writing it")
    args = parser.parse_args()
    case_set = build_case_set([p.resolve() for p in args.results])
    text = json.dumps(case_set, indent=1) + "\n"
    if args.check:
        same = CASE_SET.exists() and CASE_SET.read_text() == text
        print("case set is current" if same else "case set differs from what the inputs give")
        return 0 if same else 1
    CASE_SET.write_text(text)
    for group, by_status in case_set["counts"].items():
        print(f"{group}: " + ", ".join(f"{s} {n}" for s, n in sorted(by_status.items())))
    print(f"wrote {CASE_SET.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
