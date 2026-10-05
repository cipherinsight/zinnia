"""Output check for the SP1, RISC Zero and Cairo baselines.

For every case, run the kernel body as plain NumPy on one fixture input (the
reference), run the baseline program on the same input, and compare the public
values position by position: exactly for integers and booleans, within the
case's tolerance for floats.

Systems:
  native   the Q32 Rust program compiled for the host (no zkVM)
  sp1      the SP1 guest in SP1's executor (host `--mode execute`)
  risc0    the RISC Zero guest in RISC Zero's executor (host `--mode execute`)
  cairo    the Cairo program under `scarb execute` (driver `--mode execute`)

Usage:
  .venv/bin/python benchmarking/baselines/zkvm_common/check_outputs.py \
      [--systems native,sp1,risc0,cairo] [--cases ds1000_case0300,...] \
      [--fixtures-dir DIR] [--input B] [--report out.json]

Build the programs first (see `zkvm_common/README.md`).
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

BENCH = Path(__file__).resolve().parents[2]
DATASET = BENCH / "datasets" / "ds1000"
BASELINES = BENCH / "baselines"
CASE_SET = BENCH / "datasets" / "case_sets" / "ds1000_perf.json"
TARGET = BASELINES / "target" / "release"

# Used when the case set does not give a tolerance for a case.
DEFAULT_RTOL = 1e-6
DEFAULT_ATOL = 1e-6

SYSTEMS = {
    "native": lambda d, f, i: [str(TARGET / "ds1000-native"), "--case-dir", str(d), "--fixture", str(f), "--input", i, "--mode", "execute"],
    "sp1": lambda d, f, i: [str(BASELINES / "sp1" / "target" / "release" / "sp1-baseline"), "--case-dir", str(d),
                            "--fixture", str(f), "--input", i, "--mode", "execute"],
    "risc0": lambda d, f, i: [str(BASELINES / "risc0" / "target" / "release" / "risc0-baseline"), "--case-dir", str(d),
                              "--fixture", str(f), "--input", i, "--mode", "execute"],
    "cairo": lambda d, f, i: [sys.executable, str(BASELINES / "cairo" / "cairo_baseline.py"), "--case-dir", str(d),
                              "--fixture", str(f), "--input", i, "--mode", "execute"],
}

# Which per-case file each system needs.
PROGRAM_FILE = {"native": "zkvm.rs", "sp1": "zkvm.rs", "risc0": "zkvm.rs", "cairo": "main.cairo"}


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


# --------------------------------------------------------------------------
# NumPy reference
# --------------------------------------------------------------------------

def _identity(fn=None, **_kwargs):
    return fn if fn is not None else (lambda f: f)


def numpy_reference(case_dir: Path, fixture: dict, input_name: str) -> Expected:
    """Run the kernel body as plain NumPy and flatten its return value."""
    import zinnia

    namespace = {name: getattr(zinnia, name) for name in dir(zinnia) if not name.startswith("_")}
    namespace.update(np=np, zk_circuit=_identity, zk_chip=_identity, zk_external=_identity)
    source = (case_dir / "kernel.py").read_text().replace("from zinnia import *", "")
    exec(compile(source, str(case_dir / "kernel.py"), "exec"), namespace)

    values = fixture["inputs"][input_name]
    args = []
    for param in fixture["parameters"]:
        value = values[param["name"]]
        if param["kind"] == "array":
            dtype = {"float": np.float64, "int": np.int64, "bool": np.bool_}[param["dtype"]]
            args.append(np.array(value, dtype=dtype))
        else:
            args.append(value)
    return flatten(namespace["kernel"](*args))


def flatten(result) -> Expected:
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
            kind = arr.dtype.kind
            if kind in "fc":
                values.append(float(v))
                is_float.append(True)
            else:
                values.append(int(v))
                is_float.append(False)
    return Expected(values, is_float)


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------

def tolerances() -> dict[str, tuple[float, float]]:
    """Per-case (rtol, atol) from the case set, when it gives one."""
    if not CASE_SET.exists():
        return {}
    data = json.loads(CASE_SET.read_text())
    entries = data.get("cases", data) if isinstance(data, dict) else data
    out = {}
    if isinstance(entries, dict):
        entries = [dict(v, case=k) for k, v in entries.items()]
    for entry in entries:
        tol = entry.get("tolerance")
        if isinstance(tol, dict):
            out[entry["case"]] = (float(tol.get("rtol", DEFAULT_RTOL)), float(tol.get("atol", DEFAULT_ATOL)))
        elif isinstance(tol, (int, float)):
            out[entry["case"]] = (DEFAULT_RTOL, float(tol))
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


def run_system(system: str, case_dir: Path, fixture_path: Path, input_name: str, timeout: int) -> list:
    cmd = SYSTEMS[system](case_dir, fixture_path, input_name)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
        raise RuntimeError(f"exit {proc.returncode}: {' | '.join(tail)}")
    report = json.loads(proc.stdout.strip().splitlines()[-1])
    if not report.get("ok"):
        raise RuntimeError(f"not ok: {report.get('error')}")
    return report["publics"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--systems", default="native")
    parser.add_argument("--cases", default="", help="comma-separated case folders (default: every case with a program)")
    parser.add_argument("--fixtures-dir", type=Path, help="read <dir>/<case>/fixture.json instead of the case folder's")
    parser.add_argument("--input", default="B")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    systems = [s for s in args.systems.split(",") if s]
    unknown = set(systems) - set(SYSTEMS)
    if unknown:
        parser.error(f"unknown systems: {sorted(unknown)}")
    if args.cases:
        cases = args.cases.split(",")
    else:
        cases = sorted(d.name for d in DATASET.iterdir()
                       if any((d / PROGRAM_FILE[s]).exists() for s in systems))
    tol = tolerances()

    results: list[Result] = []
    for case in cases:
        case_dir = DATASET / case
        fixture_path = (args.fixtures_dir / case / "fixture.json") if args.fixtures_dir else case_dir / "fixture.json"
        if not fixture_path.exists():
            results += [Result(case, s, False, "no fixture") for s in systems]
            continue
        fixture = json.loads(fixture_path.read_text())
        try:
            expected = numpy_reference(case_dir, fixture, args.input)
        except Exception as exc:  # the reference itself failed: report, do not hide
            results += [Result(case, s, False, f"NumPy reference failed: {exc!r}") for s in systems]
            continue
        rtol, atol = tol.get(case, (DEFAULT_RTOL, DEFAULT_ATOL))
        for system in systems:
            if not (case_dir / PROGRAM_FILE[system]).exists():
                results.append(Result(case, system, False, f"no {PROGRAM_FILE[system]}"))
                continue
            try:
                publics = run_system(system, case_dir, fixture_path, args.input, args.timeout)
            except Exception as exc:
                results.append(Result(case, system, False, f"run failed: {exc}"))
                continue
            results.append(compare(case, system, publics, expected, rtol, atol))

    width = max((len(r.case) for r in results), default=4)
    for r in results:
        status = "ok  " if r.ok else "FAIL"
        extra = f" max|err|={r.max_abs_err:.2e}" if r.max_abs_err else ""
        mism = f" first mismatches (pos, got, want): {r.mismatches}" if r.mismatches else ""
        print(f"{status} {r.case:<{width}} {r.system:<6} {r.detail}{extra}{mism}")
    failed = [r for r in results if not r.ok]
    print(f"\n{len(results) - len(failed)} of {len(results)} checks passed")
    if args.report:
        args.report.write_text(json.dumps([r.__dict__ for r in results], indent=1))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
