"""Domain-edge regression checks: inputs at the edge of what a kernel can prove.

Each entry of `domain_edges.json` is one input (input B of the case's fixture
with some elements replaced) and the outcome every baseline must give on it:

* `reject`: no proof (the halo2 mock is unsatisfied, `nargo execute` fails,
  the zkVM program panics, Cairo's execution fails);
* `numpy`: a proof whose outputs equal the NumPy reference position by
  position (the tolerance of `hw_check.check`).

`expect` is Zinnia's outcome, which the baselines must reproduce (a baseline
proves a statement exactly when Zinnia does, on the same input). An entry may
override it per system in `expect_by_system` where a baseline knowingly
differs; the entry's `why` says why.

Systems: `halo2` (`halo2_hw mock`), `noir` (`hw_check.noir_driver prove`),
`native` (the zkVM programs run natively, `kernels-native --mode execute`),
`sp1` and `risc0` (`--mode execute` of the hosts, which run the guests in the
zkVM's executor), `cairo` (`cairo_baseline.py --mode execute`), and `zinnia`
(the kernel compiled and run on Zinnia's mock prover; a reference column that
shows whether Zinnia itself gives the expected outcome).

Usage:
    python -m hw_check.domain_edges [CASES...] [--systems halo2,noir,native,cairo]
        [--out results.json]

Exit status 0 iff every selected system gives every expected outcome.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile
from pathlib import Path

from .check import BENCH, DATASET, HALO2_BIN, case_name, case_set, compare, run_json, tolerance
from .reference import reference_outputs

EDGES = Path(__file__).with_name("domain_edges.json")
BASELINES = BENCH / "baselines"
NATIVE_BIN = BASELINES / "target" / "release" / "kernels-native"
SP1_BIN = BASELINES / "sp1" / "target" / "release" / "sp1-baseline"
RISC0_BIN = BASELINES / "risc0" / "target" / "release" / "risc0-baseline"
CAIRO_DRIVER = BASELINES / "cairo" / "cairo_baseline.py"
ALL_SYSTEMS = ("halo2", "noir", "native", "sp1", "risc0", "cairo", "zinnia")
DEFAULT_SYSTEMS = "halo2,noir,native,cairo"
BASELINE_FILE = {"halo2": "target.rs", "noir": "main.nr", "native": "zkvm.rs", "sp1": "zkvm.rs",
                 "risc0": "zkvm.rs", "cairo": "main.cairo", "zinnia": "kernel.py"}


def edge_values(fixture: dict, edge: dict) -> dict:
    """Input B with the entry's `set` applied: `{param: value}` replaces a
    whole parameter, `{param: {"<flat index>": value}}` single elements."""
    values = copy.deepcopy(fixture["inputs"]["B"])
    shapes = {p["name"]: p.get("shape", []) for p in fixture["parameters"]}
    for name, change in edge["set"].items():
        if not isinstance(change, dict):
            values[name] = change
            continue
        flat = _flatten(values[name])
        for index, value in change.items():
            flat[int(index)] = value
        values[name] = _reshape(flat, shapes[name])
    return values


def _flatten(x) -> list:
    return [y for item in x for y in _flatten(item)] if isinstance(x, list) else [x]


def _reshape(flat: list, shape: list[int]):
    if len(shape) <= 1:
        return flat
    step = len(flat) // shape[0]
    return [_reshape(flat[i * step:(i + 1) * step], shape[1:]) for i in range(shape[0])]


ZINNIA_WORKER = """
import json, sys
sys.path.insert(0, sys.argv[1])
from pathlib import Path
from benchmarking.check_outputs import mock_exposed_values
from benchmarking.harness.adapters.zinnia_common import describe_compiled, with_public_input_gaps, zinnia_publics
from benchmarking.harness.fixtures import load_fixture
from benchmarking.harness.kernels import compile_kernel
fixture = load_fixture(Path(sys.argv[2]))
compiled = compile_kernel(Path(sys.argv[3]), fixture.function, backend="halo2")
info = describe_compiled(compiled)
result = {}
for name in sys.argv[4:]:
    proof = compiled.prove(*fixture.arguments(name), backend="mock")
    if proof.proof_bytes_hex != "mock_satisfied":
        result[name] = {"ok": False, "error": str(proof.proof_bytes_hex)[:300]}
        continue
    fields = with_public_input_gaps(mock_exposed_values(proof), info["public_inputs"])
    result[name] = {"ok": True, "publics": zinnia_publics(fields, info).values}
print(json.dumps(result))
"""


def _flat_publics(values) -> list:
    return [y for v in values for y in _flat_publics(v)] if isinstance(values, (list, tuple)) else [values]


def run_zinnia(case_dir: Path, fixture_path: Path, names: list[str]) -> dict[str, dict]:
    root = BENCH.parent
    cmd = [sys.executable, "-c", ZINNIA_WORKER, str(root), str(fixture_path), str(case_dir / "kernel.py"), *names]
    result = run_json(cmd, root)
    if set(result) != set(names):
        return {name: {"ok": False, "error": f"zinnia worker: {result.get('error')}"} for name in names}
    return result


_NOIR_SET_UP: set[str] = set()  # cases compiled in this run (setup is input-independent)


def run_system(system: str, case_dir: Path, fixture_path: Path, name: str, workdir: Path) -> dict:
    common = ["--fixture", str(fixture_path), "--input", name]
    if system == "halo2":
        step = run_json([str(HALO2_BIN), "mock", "--case", str(case_dir), *common])
        return {**step, "ok": bool(step.get("ok") and step.get("satisfied"))}
    if system == "noir":
        driver = [sys.executable, "-m", "hw_check.noir_driver"]
        wd = ["--workdir", str(workdir / "noir" / case_dir.name)]
        if case_dir.name not in _NOIR_SET_UP:
            setup = run_json([*driver, "setup", "--case", str(case_dir), *wd], BASELINES)
            if not setup.get("gates"):
                return {"ok": False, "error": f"setup failed: {setup.get('error')}"}
            _NOIR_SET_UP.add(case_dir.name)
        return run_json([*driver, "prove", "--case", str(case_dir), *wd, *common], BASELINES)
    zkvm = {"native": NATIVE_BIN, "sp1": SP1_BIN, "risc0": RISC0_BIN}
    if system in zkvm:
        extra = ["--work-dir", str(workdir / system / case_dir.name)] if system != "native" else []
        return run_json([str(zkvm[system]), "--case-dir", str(case_dir), *common, "--mode", "execute", *extra])
    if system == "cairo":
        return run_json([sys.executable, str(CAIRO_DRIVER), "--case-dir", str(case_dir), *common,
                         "--mode", "execute", "--work-dir", str(workdir / "cairo" / case_dir.name)])
    raise ValueError(f"unknown system {system}")


def outcome(step: dict, expected_outputs, atol: float, rtol: float) -> str:
    """`reject`, `numpy` (accepted with NumPy's outputs) or `wrong: ...`."""
    if not step.get("ok"):
        return "reject"
    if isinstance(expected_outputs, str):
        return f"wrong: accepted where NumPy fails ({expected_outputs})"
    problems = compare(_flat_publics(step.get("publics", [])), expected_outputs, atol, rtol)
    return "numpy" if not problems else "wrong: " + "; ".join(problems[:3])


def check_case(case: str, edges: list[dict], systems: list[str], workdir: Path) -> list[dict]:
    case_dir = DATASET / case
    fixture = json.loads((case_dir / "fixture.json").read_text())
    names = [f"edge_{edge['name']}" for edge in edges]
    for name, edge in zip(names, edges):
        fixture["inputs"][name] = edge_values(fixture, edge)
    fixture_path = workdir / f"{case}.fixture.json"
    fixture_path.write_text(json.dumps(fixture))
    atol, rtol = tolerance(case_set().get(case))
    expected_outputs = {}
    for name in names:
        try:
            expected_outputs[name] = reference_outputs(case_dir, fixture, name)
        except Exception as exc:  # noqa: BLE001 - NumPy raising is an outcome
            expected_outputs[name] = f"{type(exc).__name__}: {exc}"
    present = [s for s in systems if (case_dir / BASELINE_FILE[s]).exists()]
    zinnia = run_zinnia(case_dir, fixture_path, names) if "zinnia" in present else {}
    rows = []
    for name, edge in zip(names, edges):
        row = {"case": case, "edge": edge["name"], "expect": edge["expect"], "systems": {}}
        for system in present:
            step = zinnia[name] if system == "zinnia" else run_system(system, case_dir, fixture_path, name, workdir)
            got = outcome(step, expected_outputs[name], atol, rtol)
            want = edge.get("expect_by_system", {}).get(system, edge["expect"])
            row["systems"][system] = {"got": got, "want": want, "pass": got == want,
                                      "error": step.get("error") if got == "reject" else None}
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cases", nargs="*", help="cases to check (default: every case in domain_edges.json)")
    ap.add_argument("--systems", default=DEFAULT_SYSTEMS, help=f"comma-separated subset of {','.join(ALL_SYSTEMS)}")
    ap.add_argument("--workdir", type=Path, default=Path(tempfile.gettempdir()) / "hw_domain_edges")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    table = json.loads(EDGES.read_text())["cases"]
    cases = [case_name(c) for c in a.cases] or sorted(table)
    systems = a.systems.split(",")
    unknown = set(systems) - set(ALL_SYSTEMS)
    if unknown:
        ap.error(f"unknown systems {sorted(unknown)}")
    a.workdir.mkdir(parents=True, exist_ok=True)
    rows, failures = [], 0
    for case in cases:
        for row in check_case(case, table[case], systems, a.workdir):
            rows.append(row)
            cells = []
            for system, result in row["systems"].items():
                mark = "ok" if result["pass"] else "FAIL"
                cells.append(f"{system}={result['got']}" + ("" if result["pass"] else f" ({mark}, want {result['want']})"))
                # Zinnia is the reference column: a mismatch there is reported, not counted.
                failures += system != "zinnia" and not result["pass"]
            print(f"{case} {row['edge']} [expect {row['expect']}]  " + "  ".join(cells), flush=True)
    if a.out:
        a.out.write_text(json.dumps(rows, indent=1))
    print(f"{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    sys.exit(main())
