"""Baseline programs driven through the per-system program contract.

The contract (`eval-redo/BRIEF.md`; the command line is fixed here and in
`benchmarking/README.md`): every baseline system directory
`benchmarking/baselines/<dir>/` provides an executable `bench` called as

    bench --case-dir DIR --fixture PATH --input A|B --mode setup|prove|verify
          --work-dir DIR --out PATH [--proof PATH] [--tamper] [system flags]

- `setup` (input A): build and generate keys into `--work-dir`.
- `prove` (input B): one proof written to exactly `--proof` (the file the harness
  verifies, copies for the tamper check and measures) using the setup's keys.
- `verify`: verify the proof at `--proof`; with `--tamper`, first change one
  public value of that proof (the harness passes a copy).

and writes one JSON object to `--out`:

    {"ok": bool, "error": str?, "error_class": str?,
     "publics": [kernel outputs, flattened row-major, decoded],   # prove
     "shapes": [[d, ...] per output part],                        # prove, optional
     "proof_path": str,                                           # prove
     "verified": bool,                                            # verify
     "timings_s": {"setup": s, "witness": s, "prove": s, "verify": s},
     "details": {...}}

`timings_s.prove` is the prover call with the witness or execution trace
already computed; `timings_s.witness` is witness generation or execution. A
program exits non-zero on any failure (including a rejected proof) but still
writes the JSON. `error_class` may name a failure class of the protocol (for
example `unsupported`). `halo2_hw` also accepts `--k K --lookup-bits L` in
setup (a parameter override for sensitivity runs) and reports the chosen
`k` and `lookup_bits` in `details`.
"""

from __future__ import annotations

import json
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from benchmarking.harness import paths
from benchmarking.harness.adapters.base import (Adapter, ProveResult, SetupResult,
                                                StepContext, StepFailure, VerifyResult)
from benchmarking.harness.case_sets import PerfCase
from benchmarking.harness.fixtures import SETUP_INPUT
from benchmarking.harness.outputs import LOGICAL, DecodedPublics

ENTRY_POINT_NAME = "bench"


@dataclass(frozen=True)
class ContractSystem:
    name: str
    baseline_dir: str
    #: Files in the case folder that make up this system's program for the case.
    case_files: tuple[str, ...]
    extra_args: tuple[str, ...]
    description: str


CONTRACT_SYSTEMS: dict[str, ContractSystem] = {s.name: s for s in (
    ContractSystem("halo2_hw", "halo2_hw", ("target.rs",), (),
                   "handwritten halo2-base (same prover stack as Zinnia's halo2-base backend)"),
    ContractSystem("sp1", "sp1", ("sp1.prog.rs",), (), "SP1 v6.8.1, Q32 integer guest"),
    ContractSystem("sp1_plonk", "sp1", ("sp1.prog.rs",), ("--variant", "plonk"),
                   "SP1 v6.8.1, PLONK-wrapped proof"),
    ContractSystem("risc0", "risc0", ("risc0.prog.rs",), (), "RISC Zero 3.0.3, Q32 guest"),
    ContractSystem("cairo", "cairo", ("main.cairo",), (), "Cairo / Scarb 2.20.1, Q32"),
)}

KNOWN_ERROR_CLASSES = {"unsupported", "setup_failed", "prove_failed", "witness_dependent",
                       "out_of_memory"}


class ContractProgramAdapter(Adapter):
    windows = {
        "setup": "the program's `timings_s.setup`",
        "prove_core": "the program's `timings_s.prove`: the prover call, witness or trace "
                      "already computed",
        "end_to_end": "the program's `timings_s.witness` + `timings_s.prove`",
        "verify": "the program's `timings_s.verify`: the verifier call, keys preloaded",
    }

    def __init__(self, system: ContractSystem):
        self.system = system
        self.name = system.name
        self.description = system.description

    @property
    def entry_point(self) -> Path:
        return paths.BASELINES / self.system.baseline_dir / ENTRY_POINT_NAME

    def missing_program(self, case: PerfCase) -> str | None:
        if not self.entry_point.exists():
            return f"no entry point {paths.relative_to_repo(self.entry_point)}"
        if not any((case.directory / f).exists() for f in self.system.case_files):
            return f"no {' / '.join(self.system.case_files)} in the case folder"
        return None

    def program_files(self, case: PerfCase) -> list[Path]:
        return [case.directory / f for f in self.system.case_files
                if (case.directory / f).exists()]

    def _invoke(self, ctx: StepContext, mode: str, input_name: str,
                extra: list[str]) -> tuple[dict, float, int]:
        out = ctx.work_dir / f"{mode}-{uuid.uuid4().hex[:8]}.json"
        command = [str(self.entry_point), "--case-dir", str(ctx.case.directory),
                   "--fixture", str(ctx.case.fixture_path), "--input", input_name,
                   "--mode", mode, "--work-dir", str(ctx.work_dir), "--out", str(out),
                   *self.system.extra_args, *extra]
        started = time.perf_counter()
        proc = subprocess.run(command, cwd=paths.REPO_ROOT, capture_output=True, text=True)
        wall_s = time.perf_counter() - started
        (ctx.work_dir / f"{out.stem}.log").write_text(
            f"$ {' '.join(command)}\n{proc.stdout}\n{proc.stderr}")
        if proc.returncode == -9:
            raise StepFailure("out_of_memory", f"{self.name} {mode} killed (SIGKILL)")
        if not out.exists():
            raise StepFailure("crashed", f"{self.name} {mode} exited {proc.returncode} "
                              f"without writing {out.name}: {proc.stderr[-500:]}")
        return json.loads(out.read_text()), wall_s, proc.returncode

    @staticmethod
    def _failure(response: dict, default_class: str) -> StepFailure:
        error_class = response.get("error_class")
        failure_class = error_class if error_class in KNOWN_ERROR_CLASSES else default_class
        return StepFailure(failure_class, str(response.get("error", "program reported ok=false")))

    def setup(self, ctx: StepContext) -> SetupResult:
        extra: list[str] = []
        override = ctx.options.get("halo2_params")
        if override and self.name == "halo2_hw":
            extra = ["--k", str(override["k"]), "--lookup-bits", str(override["lookup_bits"])]
        response, wall_s, _ = self._invoke(ctx, "setup", SETUP_INPUT, extra)
        if not response.get("ok"):
            raise self._failure(response, "setup_failed")
        timings = dict(response.get("timings_s", {}))
        timings["process_wall"] = wall_s
        return SetupResult(timings_s=timings, details=response.get("details", {}))

    def prove(self, ctx: StepContext) -> ProveResult:
        ctx.proof_dir.mkdir(parents=True, exist_ok=True)
        proof = ctx.proof_dir / "proof"
        response, wall_s, _ = self._invoke(ctx, "prove", ctx.input_name, ["--proof", str(proof)])
        if not response.get("ok"):
            raise self._failure(response, "prove_failed")
        timings = dict(response.get("timings_s", {}))
        if "prove" not in timings:
            raise StepFailure("harness_error", "program reported no timings_s.prove")
        if not proof.is_file():
            raise StepFailure("prove_failed", f"no proof written at {proof}")
        timings["process_wall"] = wall_s
        return ProveResult(
            prove_core_s=float(timings["prove"]),
            end_to_end_s=float(timings.get("witness", 0.0)) + float(timings["prove"]),
            proof_path=str(proof),
            proof_bytes=proof.stat().st_size,
            publics=DecodedPublics(LOGICAL, list(response.get("publics", [])),
                                   shapes=response.get("shapes")),
            timings_s=timings, details=response.get("details", {}))

    def verify(self, ctx: StepContext) -> VerifyResult:
        extra = ["--proof", str(ctx.proof_path)] + (["--tamper"] if ctx.tamper else [])
        response, wall_s, _ = self._invoke(ctx, "verify", ctx.input_name, extra)
        if "verified" not in response:
            raise StepFailure("harness_error", f"verify response has no 'verified': {response}")
        timings = dict(response.get("timings_s", {}))
        timings["process_wall"] = wall_s
        not_applicable = response.get("details", {}).get("tamper_not_applicable")
        return VerifyResult(verified=bool(response["verified"]),
                            verify_s=timings.get("verify"), not_applicable=not_applicable,
                            timings_s=timings, details=response.get("details", {}))
