"""Checks the handwritten baselines against the NumPy reference.

For every requested case and system with a baseline:

* halo2 (`target.rs`, through the `halo2_hw` binary): `mock` with input A and
  with input B (every constraint satisfied, outputs equal to NumPy), `setup`
  with A, `prove` with B (outputs equal to NumPy), `verify`, and `verify` with
  one public value tampered, which must be rejected.
* noir (`main.nr`, through `hw_check.noir_driver`): `setup`, `prove` with A and
  with B (outputs equal to NumPy), `verify`, tampered `verify` (rejected).

Keys always come from input A and proofs from input B, so a circuit whose
structure depends on the input values fails (H-11).

Outputs are compared position by position against the kernel body run as
plain NumPy (`hw_check.reference`): integers and booleans exactly, floats
within `atol + rtol * |reference|` with the case's `float_tolerance` from
`datasets/case_sets/ds1000_perf.json` (1e-6 each when the case set gives none;
`--tol CASE=ATOL` overrides the absolute part).

Usage:
    python -m hw_check.check 0309 0310 ... [--systems halo2,noir]
        [--fixture-root DIR] [--halo2-bin PATH] [--workroot DIR] [--out results.json]
    python -m hw_check.check --all

Heavy steps run one at a time across concurrent checkers (a file lock) and,
when `eval-redo/tools/memrun.sh` exists, under its 8 GB memory cap.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

from .reference import reference_outputs

BENCH = Path(__file__).resolve().parents[2]
DATASET = BENCH / "datasets" / "ds1000"
CASE_SET = BENCH / "datasets" / "case_sets" / "ds1000_perf.json"
HALO2_BIN = BENCH / "baselines" / "halo2_hw" / "target" / "release" / "halo2_hw"
MEMRUN = Path("/home/zhantong/Projects/zinnia-project/eval-redo/tools/memrun.sh")
LOCK = Path(os.environ.get("HW_CHECK_LOCK", Path(tempfile.gettempdir()) / "hw_check_heavy.lock"))
DEFAULT_ATOL = 1e-6
DEFAULT_RTOL = 1e-6


@contextmanager
def heavy_slot():
    """Serializes heavy steps across all running checkers."""
    with open(LOCK, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def run_json(cmd: list[str], cwd: Path | None = None) -> dict:
    if MEMRUN.exists():
        cmd = [str(MEMRUN), *cmd]
    env = dict(os.environ, MEMRUN_MAX=os.environ.get("MEMRUN_MAX", "8G"))
    with heavy_slot():
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, env=env)
    for line in reversed(proc.stdout.strip().splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    return {"ok": False, "error": f"exit {proc.returncode}: {(proc.stderr or proc.stdout)[-2000:]}"}


def compare(got: list, expected: list[tuple[str, object]], atol: float, rtol: float) -> list[str]:
    """Position-by-position differences, as human-readable strings."""
    if len(got) != len(expected):
        return [f"length {len(got)} != NumPy length {len(expected)}"]
    problems = []
    for i, (g, (kind, e)) in enumerate(zip(got, expected)):
        if kind == "float":
            if not isinstance(g, (int, float)) or isinstance(g, bool) or not math.isfinite(e) \
                    or abs(g - e) > atol + rtol * abs(e):
                problems.append(f"[{i}] {g!r} != {e!r}")
        elif kind == "bool":
            if bool(g) != bool(e) or not isinstance(g, (bool, int)):
                problems.append(f"[{i}] {g!r} != {e!r}")
        elif isinstance(g, bool) or g != e:
            problems.append(f"[{i}] {g!r} != {e!r}")
    return problems[:10] + ([f"... {len(problems) - 10} more"] if len(problems) > 10 else [])


class CaseCheck:
    def __init__(self, case: str, fixture_path: Path, workroot: Path, halo2_bin: Path, atol: float, rtol: float):
        self.case = case
        self.dir = DATASET / case
        self.fixture_path = fixture_path
        self.fixture = json.loads(fixture_path.read_text())
        self.work = workroot / case
        self.halo2_bin = halo2_bin
        self.atol, self.rtol = atol, rtol
        self.expected = {name: reference_outputs(self.dir, self.fixture, name) for name in ("A", "B")}

    def _outputs_ok(self, step: dict, input_name: str, problems: list[str], label: str) -> None:
        if not step.get("ok"):
            problems.append(f"{label}: {step.get('error', 'not ok')}")
            return
        diff = compare(step.get("publics", []), self.expected[input_name], self.atol, self.rtol)
        problems += [f"{label}: {d}" for d in diff]

    def _tamper(self, verify, n_publics: int, problems: list[str]) -> dict | None:
        if n_publics == 0:
            return None
        bad = verify(0)
        if bad.get("verified") is not False:
            problems.append(f"tampered public accepted: {bad}")
        return bad

    def halo2(self) -> dict:
        b = str(self.halo2_bin)
        common = ["--case", str(self.dir), "--fixture", str(self.fixture_path)]
        wd = ["--workdir", str(self.work / "halo2")]
        problems: list[str] = []
        steps = {
            "mock_A": run_json([b, "mock", *common, "--input", "A"]),
            "mock_B": run_json([b, "mock", *common, "--input", "B"]),
        }
        self._outputs_ok(steps["mock_A"], "A", problems, "mock A")
        self._outputs_ok(steps["mock_B"], "B", problems, "mock B")
        steps["setup_A"] = run_json([b, "setup", *common, "--input", "A", *wd])
        if not steps["setup_A"].get("ok"):
            problems.append(f"setup A: {steps['setup_A'].get('error')}")
        else:
            steps["prove_B"] = run_json([b, "prove", *common, "--input", "B", *wd])
            self._outputs_ok(steps["prove_B"], "B", problems, "prove B")
            if steps["prove_B"].get("ok"):
                steps["verify"] = run_json([b, "verify", "--case", str(self.dir), *wd])
                if steps["verify"].get("verified") is not True:
                    problems.append("honest proof rejected")
                n = steps["setup_A"].get("cells", {}).get("instances", 0)
                steps["verify_tampered"] = self._tamper(
                    lambda i: run_json([b, "verify", "--case", str(self.dir), *wd, "--tamper-index", str(i)]),
                    n, problems)
        return {
            "verdict": "pass" if not problems else "FAIL",
            "problems": problems,
            "k": steps.get("setup_A", {}).get("k"),
            "cells": steps.get("setup_A", {}).get("cells"),
            "steps": steps,
        }

    def noir(self) -> dict:
        py = sys.executable
        drv = [py, "-m", "hw_check.noir_driver"]
        common = ["--case", str(self.dir), "--workdir", str(self.work / "noir")]
        cwd = BENCH / "baselines"
        problems: list[str] = []
        steps = {"setup": run_json([*drv, "setup", *common], cwd)}
        if not steps["setup"].get("ok"):
            problems.append(f"setup: {steps['setup'].get('error')}")
        else:
            for name in ("A", "B"):
                steps[f"prove_{name}"] = run_json(
                    [*drv, "prove", *common, "--fixture", str(self.fixture_path), "--input", name], cwd)
                self._outputs_ok(steps[f"prove_{name}"], name, problems, f"prove {name}")
            if steps["prove_B"].get("ok"):
                steps["verify"] = run_json([*drv, "verify", *common], cwd)
                if steps["verify"].get("verified") is not True:
                    problems.append("honest proof rejected")
                n = len(Path(steps["prove_B"]["public_inputs_path"]).read_bytes()) // 32
                steps["verify_tampered"] = self._tamper(
                    lambda i: run_json([*drv, "verify", *common, "--tamper-index", str(i)], cwd), n, problems)
        return {
            "verdict": "pass" if not problems else "FAIL",
            "problems": problems,
            "gates": steps["setup"].get("gates"),
            "steps": steps,
        }


def case_set() -> dict[str, dict]:
    if not CASE_SET.exists():
        return {}
    return {c["case"]: c for c in json.loads(CASE_SET.read_text())["cases"]}


def tolerance(entry: dict | None) -> tuple[float, float]:
    tol = (entry or {}).get("float_tolerance") or {}
    return tol.get("atol", DEFAULT_ATOL), tol.get("rtol", DEFAULT_RTOL)


def case_name(spec: str) -> str:
    return spec if spec.startswith("ds1000_case") else f"ds1000_case{spec}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cases", nargs="*")
    ap.add_argument("--all", action="store_true", help="every case with a target.rs or main.nr")
    ap.add_argument("--perf", action="store_true", help="every case whose case-set status is perf")
    ap.add_argument("--systems", default="halo2,noir")
    ap.add_argument("--fixture-root", type=Path, help="<root>/<case>/fixture.json instead of the case folder")
    ap.add_argument("--halo2-bin", type=Path, default=HALO2_BIN)
    ap.add_argument("--workroot", type=Path, default=Path(tempfile.gettempdir()) / "hw_check_work")
    ap.add_argument("--tol", action="append", default=[], help="CASE=ATOL (rtol stays 1e-6)")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    statuses = case_set()
    if a.all:
        cases = sorted(d.name for d in DATASET.iterdir() if (d / "target.rs").exists() or (d / "main.nr").exists())
    elif a.perf:
        cases = sorted(c for c, e in statuses.items() if e["status"] == "perf")
    else:
        cases = [case_name(c) for c in a.cases]
    atol_override = {case_name(k): float(v) for k, v in (t.split("=") for t in a.tol)}
    systems = a.systems.split(",")

    results, failed = {}, 0
    for case in cases:
        fixture = (a.fixture_root / case / "fixture.json") if a.fixture_root else (DATASET / case / "fixture.json")
        entry: dict = {}
        if not fixture.exists():
            entry["error"] = f"no fixture at {fixture}"
        else:
            try:
                atol, rtol = tolerance(statuses.get(case))
                cc = CaseCheck(case, fixture, a.workroot, a.halo2_bin, atol_override.get(case, atol), rtol)
                if "halo2" in systems and (cc.dir / "target.rs").exists():
                    entry["halo2"] = cc.halo2()
                if "noir" in systems and (cc.dir / "main.nr").exists():
                    entry["noir"] = cc.noir()
            except Exception as e:  # a crashing reference is a finding, not a checker crash
                entry["error"] = f"{type(e).__name__}: {e}"
        if statuses.get(case, {}).get("status") not in (None, "perf"):
            entry["status"] = statuses[case]["status"]
        results[case] = entry
        line = [case] + ([f"[{entry['status']}]"] if "status" in entry else [])
        for s in ("halo2", "noir"):
            if s in entry:
                size = entry[s].get("cells") if s == "halo2" else entry[s].get("gates")
                line.append(f"{s}={entry[s]['verdict']} ({size})")
                if entry[s]["verdict"] != "pass":
                    failed += 1
                    line.append("; ".join(entry[s]["problems"][:3]))
        if "error" in entry:
            failed += 1
            line.append(f"ERROR {entry['error']}")
        print("  ".join(str(x) for x in line), flush=True)
    if a.out:
        a.out.write_text(json.dumps(results, indent=1))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
