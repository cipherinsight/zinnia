"""Checks the handwritten baselines against the NumPy reference.

Cases are DS-1000 cases and NPBench/Pythran entries (`hw_check.entries`: an
entry's files are `target.<function>.rs` and `main.<function>.nr` in its
program folder, its fixture and tolerance are the case set's, and a function
returning `None` outputs the arguments the case set lists as written).

For every requested case and system with a baseline:

* halo2 (`target.rs`, through the `halo2_hw` binary): `mock` with input A and
  with input B (every constraint satisfied, outputs equal to NumPy), `setup`
  with A, `prove` with B (outputs equal to NumPy), `verify`, and `verify` with
  one public value tampered, which must be rejected.
* noir (`main.nr`, through `hw_check.noir_driver`): `setup`, `prove` with A and
  with B (outputs equal to NumPy), `verify`, tampered `verify` (rejected).

With `--no-prove` nothing generates keys or proofs: halo2 runs the two mock
runs and `stats` (cells at the size Zinnia's rule chooses), Noir runs `stats`
(gates) and `execute` with A and with B (witness solving, outputs equal to
NumPy). The out-of-domain step below runs as a mock (halo2) or an `execute`
(Noir) in both forms; the tampered-public step needs a proof and is skipped.

Keys always come from input A and proofs from input B, so a circuit whose
structure depends on the input values fails (H-11).

Input domain: one input element the kernel reads (the first, in parameter
order, whose change changes NumPy's output on input B, else A;
`CaseCheck.read_input`) is moved out of its dtype's domain (halo2: `mock
--tamper-input`, a float to Q32 code 2^63 or an integer to 2^64 + v; Noir:
`prove --tamper-input`, the field value + 2^64); both systems must reject it
(the same input untampered is accepted by the A/B steps). When no element
changes the output (a degenerate kernel), the first non-Boolean input element
is used; a kernel without one skips the step. The halo2 mock also reports `redundant_input_checks`, which
must be 0 (each input checked once); literal baselines are exempt, since
they re-check an operand at every comparison (LITERAL_BASELINES.md L5).

Outputs are compared position by position against the kernel body run as
plain NumPy (`hw_check.reference`): integers and booleans exactly, floats
within `atol + rtol * |reference|` with the case's `float_tolerance` from
its case set (`datasets/case_sets/ds1000_perf.json`,
`npbench_pythran_perf.json`; 1e-6 each when the case set gives none;
`--tol CASE=ATOL` overrides the absolute part).

Usage:
    python -m hw_check.check 0309 0310 pythran_lu__factorMatrix0 ... [--systems halo2,noir]
        [--fixture-root DIR] [--halo2-bin PATH] [--workroot DIR] [--out results.json]
    python -m hw_check.check --all [--npb]     every DS-1000 case (NPBench/Pythran entry) with a baseline
    python -m hw_check.check --perf [--npb]    every case of the performance set
    python -m hw_check.check ... --no-prove     mock / execute and counts only

Heavy steps run one at a time across concurrent checkers (a file lock) and,
when `eval-redo/tools/memrun.sh` exists, under its 8 GB memory cap.
"""
from __future__ import annotations

import argparse
import dataclasses
import fcntl
import json
import math
import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .entries import BENCH, VARIANT, Entry, case_name, ds1000_entries, npb_entries, resolve, with_baselines
from .reference import call_outputs, entry_outputs, fixture_arguments, flatten_output, load_kernel

# Each variant is built into its own target directory (`halo2_hw/build.rs`).
HALO2_BIN = (BENCH / "baselines" / "halo2_hw" / ("target-" + VARIANT if VARIANT else "target")
             / "release" / "halo2_hw")
MEMRUN = Path("/home/zhantong/Projects/zinnia-project/eval-redo/tools/memrun.sh")
LOCK = Path(os.environ.get("HW_CHECK_LOCK", Path(tempfile.gettempdir()) / "hw_check_heavy.lock"))


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
            if not isinstance(g, (int, float)) or isinstance(g, bool) or not math.isfinite(g) or not math.isfinite(e) \
                    or abs(g - e) > atol + rtol * abs(e):
                problems.append(f"[{i}] {g!r} != {e!r}")
        elif kind == "bool":
            if not isinstance(g, (bool, int)) or g not in (0, 1) or bool(g) != bool(e):
                problems.append(f"[{i}] {g!r} != {e!r}")
        elif isinstance(g, bool) or g != e:
            problems.append(f"[{i}] {g!r} != {e!r}")
    return problems[:10] + ([f"... {len(problems) - 10} more"] if len(problems) > 10 else [])


class CaseCheck:
    def __init__(self, entry: Entry, workroot: Path, halo2_bin: Path, atol: float, rtol: float,
                 prove: bool = True):
        self.entry = entry
        self.case = entry.case
        self.dir = entry.directory
        self.fixture_path = entry.fixture
        self.fixture = json.loads(entry.fixture.read_text())
        self.work = workroot / entry.case
        self.halo2_bin = halo2_bin
        self.atol, self.rtol = atol, rtol
        self.prove = prove
        self.expected = {name: entry_outputs(entry, self.fixture, name) for name in ("A", "B")}

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

    def read_input(self) -> tuple[str, int, str] | None:
        """An input element the kernel reads, as (parameter, flat index, input
        set) for the out-of-domain step: the first element (parameter order;
        each parameter's first, last and middle element before the others)
        whose change changes NumPy's output, on input B, else on input A.
        Boolean parameters are skipped (their out-of-domain value has no
        fixture encoding). The index is a leaf index of the fixture value: a
        complex element is two leaves, and its real part is the one moved."""
        kernel = load_kernel(self.entry.kernel, self.entry.kernel_function)
        mutated = self.entry.mutated_outputs
        params = self.fixture["parameters"]
        sizes = [math.prod(p["shape"]) if p["kind"] == "array" else 1 for p in params]
        order = [(k, i) for k, n in enumerate(sizes) for i in dict.fromkeys([0, n - 1, n // 2])]
        order += [(k, i) for k, n in enumerate(sizes) for i in range(n) if i not in (0, n - 1, n // 2)]
        for input_name in ("B", "A"):
            args = fixture_arguments(self.fixture, input_name)
            fresh = lambda: [a.copy() if isinstance(a, np.ndarray) else a for a in args]
            base = flatten_output(call_outputs(kernel, fresh(), mutated))
            for k, index in order:
                if params[k]["dtype"] == "bool":
                    continue
                for delta in (1000, -1000, 1, -1):
                    trial = fresh()
                    if params[k]["kind"] == "scalar":
                        trial[k] = trial[k] + delta
                    else:
                        trial[k].reshape(-1)[index] += delta
                    try:
                        out = flatten_output(call_outputs(kernel, trial, mutated))
                    except Exception:
                        continue
                    if out != base:
                        leaf = 2 * index if params[k]["dtype"] == "complex" else index
                        return params[k]["name"], leaf, input_name
        # No element changes the output (a degenerate kernel, whose outputs are fixed by the
        # types and constants): its inputs are still declared, so the first non-Boolean one
        # must still be range-checked.
        for k, index in order:
            if params[k]["dtype"] != "bool":
                leaf = 2 * index if params[k]["dtype"] == "complex" else index
                return params[k]["name"], leaf, "B"
        return None

    def tamper_target(self) -> tuple[str, int, str] | None:
        if not hasattr(self, "_target"):
            self._target = self.read_input()
        return self._target

    def halo2(self) -> dict:
        b = str(self.halo2_bin)
        case = self.entry.program_args()
        common = [*case, "--fixture", str(self.fixture_path)]
        wd = ["--workdir", str(self.work / "halo2")]
        problems: list[str] = []
        steps = {
            "mock_A": run_json([b, "mock", *common, "--input", "A"]),
            "mock_B": run_json([b, "mock", *common, "--input", "B"]),
        }
        self._outputs_ok(steps["mock_A"], "A", problems, "mock A")
        self._outputs_ok(steps["mock_B"], "B", problems, "mock B")
        if not self.prove:
            steps["stats_A"] = run_json([b, "stats", *common, "--input", "A"])
            if not steps["stats_A"].get("ok"):
                problems.append(f"stats A: {steps['stats_A'].get('error')}")
        else:
            steps["setup_A"] = run_json([b, "setup", *common, "--input", "A", *wd])
            if not steps["setup_A"].get("ok"):
                problems.append(f"setup A: {steps['setup_A'].get('error')}")
            else:
                steps["prove_B"] = run_json([b, "prove", *common, "--input", "B", *wd])
                self._outputs_ok(steps["prove_B"], "B", problems, "prove B")
                if steps["prove_B"].get("ok"):
                    steps["verify"] = run_json([b, "verify", *case, *wd])
                    if steps["verify"].get("verified") is not True:
                        problems.append("honest proof rejected")
                    n = steps["setup_A"].get("cells", {}).get("instances", 0)
                    steps["verify_tampered"] = self._tamper(
                        lambda i: run_json([b, "verify", *case, *wd, "--tamper-index", str(i)]),
                        n, problems)
        # A literal baseline bounds each operand at every source comparison
        # (LITERAL_BASELINES.md L5, L6), so repeated input checks are expected there.
        if steps["mock_B"].get("redundant_input_checks") and VARIANT != "literal":
            problems.append(f"{steps['mock_B']['redundant_input_checks']} inputs checked twice")
        target = self.tamper_target()
        if target is None:
            steps["out_of_domain"] = "skipped: no non-Boolean input"
        else:
            name, index, input_name = target
            spec = f"{name}:{index}"
            steps["mock_out_of_domain"] = run_json(
                [b, "mock", *common, "--input", input_name, "--tamper-input", spec])
            if steps["mock_out_of_domain"].get("satisfied") is not False:
                problems.append(f"out-of-domain input {spec} accepted")
        sized = steps.get("setup_A") or steps.get("stats_A") or {}
        return {
            "verdict": "pass" if not problems else "FAIL",
            "problems": problems,
            "k": sized.get("k"),
            "cells": sized.get("cells"),
            "steps": steps,
        }

    def noir(self) -> dict:
        py = sys.executable
        drv = [py, "-m", "hw_check.noir_driver"]
        common = [*self.entry.program_args(), "--workdir", str(self.work / "noir")]
        cwd = BENCH / "baselines"
        problems: list[str] = []
        sizing, solving = ("setup", "prove") if self.prove else ("stats", "execute")
        steps = {sizing: run_json([*drv, sizing, *common], cwd)}
        if not steps[sizing].get("ok"):
            problems.append(f"{sizing}: {steps[sizing].get('error')}")
        else:
            for name in ("A", "B"):
                steps[f"{solving}_{name}"] = run_json(
                    [*drv, solving, *common, "--fixture", str(self.fixture_path), "--input", name], cwd)
                self._outputs_ok(steps[f"{solving}_{name}"], name, problems, f"{solving} {name}")
            if self.prove and steps["prove_B"].get("ok"):
                steps["verify"] = run_json([*drv, "verify", *common], cwd)
                if steps["verify"].get("verified") is not True:
                    problems.append("honest proof rejected")
                n = len(Path(steps["prove_B"]["public_inputs_path"]).read_bytes()) // 32
                steps["verify_tampered"] = self._tamper(
                    lambda i: run_json([*drv, "verify", *common, "--tamper-index", str(i)], cwd), n, problems)
            target = self.tamper_target()
            if target is None:
                steps["out_of_domain"] = "skipped: no non-Boolean input"
            else:
                name, index, input_name = target
                spec = f"{name}:{index}"
                steps[f"{solving}_out_of_domain"] = run_json(
                    [*drv, solving, *common, "--fixture", str(self.fixture_path), "--input", input_name,
                     "--tamper-input", spec], cwd)
                if steps[f"{solving}_out_of_domain"].get("ok") is not False:
                    problems.append(f"out-of-domain input {spec} accepted")
        return {
            "verdict": "pass" if not problems else "FAIL",
            "problems": problems,
            "gates": steps[sizing].get("gates"),
            "steps": steps,
        }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cases", nargs="*")
    ap.add_argument("--all", action="store_true", help="every case with a target.rs or main.nr")
    ap.add_argument("--perf", action="store_true", help="every case whose case-set status is perf")
    ap.add_argument("--npb", action="store_true",
                    help="--all / --perf over the NPBench/Pythran entries instead of DS-1000")
    ap.add_argument("--systems", default="halo2,noir")
    ap.add_argument("--fixture-root", type=Path,
                    help="<root>/<case>/<fixture file name> instead of the case's fixture")
    ap.add_argument("--halo2-bin", type=Path, default=HALO2_BIN)
    ap.add_argument("--workroot", type=Path, default=Path(tempfile.gettempdir()) / "hw_check_work")
    ap.add_argument("--tol", action="append", default=[], help="CASE=ATOL (rtol stays 1e-6)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--no-prove", action="store_true", help="mock / execute and counts only: no keys, no proofs")
    a = ap.parse_args()

    universe = npb_entries() if a.npb else ds1000_entries()
    if a.all:
        entries = with_baselines(universe)
    elif a.perf:
        entries = [e for _, e in sorted(universe.items()) if e.status == "perf"]
    else:
        entries = [resolve(c) for c in a.cases]
    atol_override = {case_name(k): float(v) for k, v in (t.split("=") for t in a.tol)}
    systems = a.systems.split(",")
    if not systems or any(s not in ("halo2", "noir") for s in systems):
        ap.error("--systems must contain halo2 and/or noir")

    results, failed = {}, 0
    for entry in entries:
        case = entry.case
        if a.fixture_root:
            entry = dataclasses.replace(entry, fixture=a.fixture_root / case / entry.fixture.name)
        result: dict = {}
        for system, path in (("halo2", entry.target_rs), ("noir", entry.main_nr)):
            if system in systems and not path.exists():
                result[system] = {"verdict": "FAIL", "problems": [f"missing baseline: {path}"]}
        if not entry.fixture.exists():
            result["error"] = f"no fixture at {entry.fixture}"
        else:
            try:
                cc = CaseCheck(entry, a.workroot, a.halo2_bin, atol_override.get(case, entry.atol), entry.rtol,
                               prove=not a.no_prove)
                if "halo2" in systems and entry.target_rs.exists():
                    result["halo2"] = cc.halo2()
                if "noir" in systems and entry.main_nr.exists():
                    result["noir"] = cc.noir()
            except Exception as e:  # a crashing reference is a finding, not a checker crash
                result["error"] = f"{type(e).__name__}: {e}"
        if entry.status not in (None, "perf"):
            result["status"] = entry.status
        results[case] = result
        line = [case] + ([f"[{result['status']}]"] if "status" in result else [])
        for s in ("halo2", "noir"):
            if s in result:
                size = result[s].get("cells") if s == "halo2" else result[s].get("gates")
                line.append(f"{s}={result[s]['verdict']} ({size})")
                if result[s]["verdict"] != "pass":
                    failed += 1
                    line.append("; ".join(result[s]["problems"][:3]))
        if "error" in result:
            failed += 1
            line.append(f"ERROR {result['error']}")
        print("  ".join(str(x) for x in line), flush=True)
    if a.out:
        a.out.write_text(json.dumps(results, indent=1))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
