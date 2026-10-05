"""Output check for the SP1, RISC Zero and Cairo baselines.

For every case, run the kernel as plain NumPy on one fixture input (the
reference), run the baseline program on the same input, and compare the public
values position by position: exactly for integers and booleans, within the
case's tolerance for floats (complex values as real, imaginary pairs).

Cases:
  fixed-shape   datasets/ds1000/<case> (zkvm.rs and main.cairo) and
                zkvm_common/calibration/<name>
  shape-generic datasets/{npbench,pythran}/<program>, one case per entry
                function (`pub mod <function>` in zkvm.rs); Rust systems only

Systems:
  native   the Q32 Rust program compiled for the host (no zkVM)
  sp1      the SP1 guest in SP1's executor (host `--mode execute`)
  risc0    the RISC Zero guest in RISC Zero's executor (host `--mode execute`)
  cairo    the Cairo program under `scarb execute` (driver `--mode execute`)

An entry function that returns None has as outputs the array arguments it
writes, in parameter order; its program declares them on a line
`// outputs: mutated A, B` inside the entry function's module.

A fixture whose notes flag it as unusable under Q32 (values of magnitude 2^31
or more, or nonzero inputs below 2^-32; see datasets/tools/acceptance_fixtures.py)
is not checked: its cases are reported as skipped, with the note.

Usage:
  .venv/bin/python benchmarking/baselines/zkvm_common/check_outputs.py \
      [--systems native,sp1,risc0,cairo] [--cases ds1000_case0300,pythran_lu/factorMatrix0,...] \
      [--fixtures-dir DIR] [--dataset-dir DIR] [--input B] [--report out.json]

Build the programs first (see `zkvm_common/README.md`).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

BENCH = Path(__file__).resolve().parents[2]
DATASETS = BENCH / "datasets"
BASELINES = BENCH / "baselines"
CALIBRATION = BASELINES / "zkvm_common" / "calibration"
CASE_SET = DATASETS / "case_sets" / "ds1000_perf.json"
ACCEPTANCE_SET = DATASETS / "case_sets" / "acceptance.json"
TARGET = BASELINES / "target" / "release"

# Used when the case set does not give a tolerance for a case.
DEFAULT_RTOL = 1e-6
DEFAULT_ATOL = 1e-6

# Phrases by which a fixture's notes flag it as unusable under Q32.
Q32_UNUSABLE = ("outside the Q32 range", "Q32 rounds them to 0")

HOSTS = {
    "native": [str(TARGET / "kernels-native")],
    "sp1": [str(BASELINES / "sp1" / "target" / "release" / "sp1-baseline")],
    "risc0": [str(BASELINES / "risc0" / "target" / "release" / "risc0-baseline")],
    "cairo": [sys.executable, str(BASELINES / "cairo" / "cairo_baseline.py")],
}
RUST_SYSTEMS = {"native", "sp1", "risc0"}

ENTRY = re.compile(r"^pub mod (\w+)\s*\{", re.M)
MUTATED = re.compile(r"^\s*//\s*outputs:\s*mutated\s+(.+?)\s*$", re.M)


@dataclass
class Case:
    name: str                 # fixed: folder name; shape-generic: "<program>/<function>"
    folder: Path
    function: str | None      # entry function of a shape-generic program
    mutated: list[str] = field(default_factory=list)

    def has_program(self, system: str) -> bool:
        if self.function is not None:
            return system in RUST_SYSTEMS
        return (self.folder / ("main.cairo" if system == "cairo" else "zkvm.rs")).exists()


@dataclass
class Expected:
    values: list
    is_float: list[bool]


@dataclass
class Result:
    case: str
    system: str
    ok: bool
    detail: str = ""
    max_abs_err: float = 0.0
    mismatches: list = field(default_factory=list)
    skipped: bool = False
    execution: dict = field(default_factory=dict)  # the host's execution sizes (cycles, steps)


# --------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------

def discover_cases() -> dict[str, Case]:
    cases: dict[str, Case] = {}
    for root in (DATASETS / "ds1000", CALIBRATION):
        for d in sorted(root.iterdir()):
            if (d / "zkvm.rs").exists() or (d / "main.cairo").exists():
                cases[d.name] = Case(d.name, d, None)
    for root in (DATASETS / "npbench", DATASETS / "pythran"):
        for d in sorted(root.iterdir()):
            if not (d / "zkvm.rs").exists():
                continue
            text = (d / "zkvm.rs").read_text()
            entries = list(ENTRY.finditer(text))
            for i, entry in enumerate(entries):
                body = text[entry.end(): entries[i + 1].start() if i + 1 < len(entries) else len(text)]
                match = MUTATED.search(body)
                mutated = [n.strip() for n in match.group(1).split(",")] if match else []
                name = f"{d.name}/{entry.group(1)}"
                cases[name] = Case(name, d, entry.group(1), mutated)
    return cases


def fixture_path(case: Case, fixtures_dir: Path | None, dataset_dir: Path | None) -> Path:
    names = ([f"fixture.{case.function}.json", f"fixture_{case.function}.json"] if case.function else [])
    names.append("fixture.json")
    folders = [d / case.folder.name for d in (fixtures_dir, dataset_dir) if d] + [case.folder]
    for folder in folders:
        for name in names:
            if (folder / name).exists():
                return folder / name
    return case.folder / "fixture.json"


# --------------------------------------------------------------------------
# NumPy reference
# --------------------------------------------------------------------------

def _identity(fn=None, **_kwargs):
    return fn if fn is not None else (lambda f: f)


def _requires(*_args, **_kwargs):
    return lambda f: f


def to_numpy(value, param: dict):
    if param["kind"] == "tuple":
        return tuple(to_numpy(v, item) for v, item in zip(value, param["items"]))
    dtype = param["dtype"]
    if dtype == "complex":
        pairs = np.asarray(value, dtype=np.float64)
        value = pairs[..., 0] + 1j * pairs[..., 1]
        return np.asarray(value) if param["kind"] == "array" else complex(value)
    if param["kind"] == "array":
        return np.array(value, dtype={"float": np.float64, "int": np.int64, "bool": np.bool_}[dtype])
    return value


def numpy_reference(case: Case, kernel_dir: Path, fixture: dict, input_name: str) -> Expected:
    """Run the kernel as plain NumPy and flatten its outputs."""
    import zinnia

    namespace = {name: getattr(zinnia, name) for name in dir(zinnia) if not name.startswith("_")}
    namespace.update(np=np, zk_circuit=_identity, zk_chip=_identity, zk_external=_identity, requires=_requires)
    source = (kernel_dir / "kernel.py").read_text().replace("from zinnia import *", "")
    exec(compile(source, str(kernel_dir / "kernel.py"), "exec"), namespace)

    values = fixture["inputs"][input_name]
    args = {p["name"]: to_numpy(values[p["name"]], p) for p in fixture["parameters"]}
    result = namespace[case.function or fixture.get("function", "kernel")](*args.values())
    if case.mutated:
        if result is not None:
            raise ValueError("an entry function with mutated outputs returned a value")
        result = tuple(args[name] for name in case.mutated)
    return flatten(result)


def flatten(result) -> Expected:
    if result is None:
        return Expected([], [])
    if isinstance(result, tuple):
        parts = result
    elif isinstance(result, list) and any(isinstance(p, np.ndarray) for p in result):
        parts = tuple(result)
    else:
        parts = (result,)
    values, is_float = [], []
    for part in parts:
        arr = np.asarray(part)
        for v in arr.ravel():
            if arr.dtype.kind == "c":
                values += [float(v.real), float(v.imag)]
                is_float += [True, True]
            elif arr.dtype.kind == "f":
                values.append(float(v))
                is_float.append(True)
            else:
                values.append(int(v))
                is_float.append(False)
    return Expected(values, is_float)


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------

def tolerances(case_set: Path, acceptance_set: Path) -> dict[str, tuple[float, float]]:
    """Per-case (rtol, atol), where the case sets give a `float_tolerance`: the
    performance set for fixed-shape cases, the acceptance set's NPBench and
    Pythran entries for shape-generic ones (`<program>/<function>`)."""
    out = {}
    entries = []
    if case_set.exists():
        entries += [(entry["case"], entry) for entry in json.loads(case_set.read_text())["cases"]]
    if acceptance_set.exists():
        entries += [(f"{entry['program']}/{entry['function']}", entry)
                    for entry in json.loads(acceptance_set.read_text())["entries"]
                    if entry["dataset"] in ("npbench", "pythran")]
    for case, entry in entries:
        tol = entry.get("float_tolerance")
        if tol:
            out[case] = (float(tol["rtol"]), float(tol["atol"]))
    return out


def compare(case: str, system: str, publics: list, expected: Expected, rtol: float, atol: float) -> Result:
    if len(publics) != len(expected.values):
        return Result(case, system, False, f"{len(publics)} publics, NumPy has {len(expected.values)}")
    mismatches, max_err = [], 0.0
    for pos, (got, want, is_float) in enumerate(zip(publics, expected.values, expected.is_float)):
        got = float(got) if is_float else int(got)
        if is_float:
            err = abs(got - want)
            max_err = max(max_err, err)
            good = math.isfinite(got) and err <= atol + rtol * abs(want)
        else:
            good = got == want
        if not good:
            mismatches.append((pos, got, want))
    detail = f"{len(mismatches)} of {len(publics)} differ" if mismatches else f"{len(publics)} values"
    return Result(case, system, not mismatches, detail, max_err, mismatches[:5])


# The execution-size fields of a host report, kept in the check report.
EXECUTION_FIELDS = ("cycles", "user_cycles", "total_cycles", "segments", "resources")


def run_system(system: str, case: Case, fixture_path: Path, input_name: str, timeout: int) -> dict:
    cmd = HOSTS[system] + ["--case-dir", str(case.folder), "--fixture", str(fixture_path),
                           "--input", input_name, "--mode", "execute"]
    if case.function:
        cmd += ["--function", case.function]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    lines = proc.stdout.strip().splitlines()
    report = json.loads(lines[-1]) if lines else {}
    if proc.returncode != 0 or not report.get("ok"):
        detail = report.get("error") or " | ".join((proc.stderr or proc.stdout).strip().splitlines()[-3:])
        raise RuntimeError(f"exit {proc.returncode}: {detail}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--systems", default="native")
    parser.add_argument("--cases", default="", help="comma-separated cases (default: every case with a program)")
    parser.add_argument("--fixtures-dir", type=Path,
                        help="read <dir>/<folder>/fixture[_<function>].json, where it exists, first")
    parser.add_argument("--dataset-dir", type=Path,
                        help="take kernel.py and fixtures from <dir>/<folder>/ where they exist (the programs "
                             "still come from this tree)")
    parser.add_argument("--input", default="B")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--case-set", type=Path, default=CASE_SET,
                        help="case set giving each case's float_tolerance (default: datasets/case_sets/ds1000_perf.json)")
    parser.add_argument("--acceptance-set", type=Path, default=ACCEPTANCE_SET,
                        help="case set giving each entry function's float_tolerance "
                             "(default: datasets/case_sets/acceptance.json)")
    args = parser.parse_args()

    systems = [s for s in args.systems.split(",") if s]
    unknown = set(systems) - set(HOSTS)
    if unknown:
        parser.error(f"unknown systems: {sorted(unknown)}")
    all_cases = discover_cases()
    if args.cases:
        missing = [c for c in args.cases.split(",") if c not in all_cases]
        if missing:
            parser.error(f"unknown cases: {missing}")
        cases = [all_cases[c] for c in args.cases.split(",")]
    else:
        cases = [c for c in all_cases.values() if any(c.has_program(s) for s in systems)]
    tol = tolerances(args.case_set, args.acceptance_set)

    results: list[Result] = []
    for case in cases:
        kernel_dir = case.folder
        if args.dataset_dir and (args.dataset_dir / case.folder.name / "kernel.py").exists():
            kernel_dir = args.dataset_dir / case.folder.name
        path = fixture_path(case, args.fixtures_dir, args.dataset_dir)
        case_systems = [s for s in systems if case.has_program(s)]
        if not path.exists():
            results += [Result(case.name, s, False, "no fixture") for s in case_systems]
            continue
        fixture = json.loads(path.read_text())
        flags = [n.strip() + "." for n in fixture.get("notes", "").split(".") if any(f in n for f in Q32_UNUSABLE)]
        if flags:
            results += [Result(case.name, s, True, "skipped, fixture unusable under Q32: " + " ".join(flags),
                               skipped=True) for s in case_systems]
            continue
        try:
            expected = numpy_reference(case, kernel_dir, fixture, args.input)
        except Exception as exc:  # the reference itself failed: report, do not hide
            results += [Result(case.name, s, False, f"NumPy reference failed: {exc!r}") for s in case_systems]
            continue
        rtol, atol = tol.get(case.name, (DEFAULT_RTOL, DEFAULT_ATOL))
        for system in case_systems:
            try:
                report = run_system(system, case, path, args.input, args.timeout)
            except Exception as exc:
                results.append(Result(case.name, system, False, f"run failed: {exc}"))
                continue
            result = compare(case.name, system, report["publics"], expected, rtol, atol)
            result.execution = {k: report[k] for k in EXECUTION_FIELDS if report.get(k) is not None}
            results.append(result)

    width = max((len(r.case) for r in results), default=4)
    for r in results:
        status = "skip" if r.skipped else "ok  " if r.ok else "FAIL"
        extra = f" max|err|={r.max_abs_err:.2e}" if r.max_abs_err else ""
        mism = f" first mismatches (pos, got, want): {r.mismatches}" if r.mismatches else ""
        print(f"{status} {r.case:<{width}} {r.system:<6} {r.detail}{extra}{mism}")
    failed = [r for r in results if not r.ok]
    skipped = sum(r.skipped for r in results)
    print(f"\n{len(results) - len(failed) - skipped} of {len(results)} checks passed, {skipped} skipped")
    if args.report:
        args.report.write_text(json.dumps([r.__dict__ for r in results], indent=1))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
