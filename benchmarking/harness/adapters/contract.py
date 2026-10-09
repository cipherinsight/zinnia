"""Baseline programs driven through the per-system program contract.

Each baseline workstream ships one program per system that takes the case
folder, the fixture, the input name and a mode (`setup`, `prove`, `verify`),
writes one JSON object (`ok`, `publics` decoded in kernel output order,
`proof_path`, `timings_s`, and `verified` in verify mode) and exits non-zero on
any failure. Two command-line styles exist:

    halo2_hw <mode> --case DIR --fixture F --input A|B --workdir W --out J
             [--k K --lookup-bits L] [--tamper-index I]
    <host>   --case-dir DIR [--function F] --mode <mode> --work-dir W --fixture F
             --input A|B --out J [--proof-mode M] [--tamper]     (SP1, RISC Zero, Cairo)

`--function` names the entry function of a shape-generic case (NPBench,
Pythran: one folder, one program per entry function); `halo2_hw` takes it in
both styles' spelling, `--case DIR --function F`, and its circuit file is per
entry function (`target.<function>.rs`).

Every program keeps keys and the proof in its work directory and overwrites the
proof on each prove. So the harness gives setup its own directory and every
prove a fresh copy of it (`cp --reflink=auto`) inside the sample's proof
directory; that copy is what the sample's verification and the tamper check
(on a copy of the whole sample directory) read.

The clocks are sums of the program's own step timings, chosen per system
(`Program`): the harness decides what is inside each window, the program
only times its steps.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from benchmarking.harness import paths
from benchmarking.harness.adapters.base import (Adapter, ProveResult, SetupResult,
                                                StepContext, StepFailure, VerifyResult)
from benchmarking.harness.case_sets import PerfCase
from benchmarking.harness.fixtures import SETUP_INPUT
from benchmarking.harness.outputs import LOGICAL, DecodedPublics

HALO2_HW_STYLE = "halo2_hw"
HOST_STYLE = "host"

#: Messages with which a program refuses to tamper because the proof has no
#: public value (every output is a constant of the circuit).
NO_PUBLIC_VALUE_MESSAGES = ("outside the 0-cell instance", "no public values to tamper with")


@dataclass(frozen=True)
class Program:
    name: str
    description: str
    #: The per-case program file in the case folder.
    case_file: str
    #: The command, relative to `paths.BASELINES`; a leading `python` means
    #: the harness's own interpreter.
    command: tuple[str, ...]
    style: str
    extra_args: tuple[str, ...]
    #: Step timings (keys of the program's `timings_s`) summed into each clock.
    prove_core: tuple[str, ...]
    end_to_end: tuple[str, ...]
    verify: tuple[str, ...]
    windows: dict = field(default_factory=dict)
    #: In a shape-generic folder, `case_file` is per entry function
    #: (`target.rs` -> `target.<function>.rs`) rather than one for the program.
    file_per_function: bool = False

    def case_file_path(self, case: PerfCase) -> Path:
        if self.file_per_function:
            return case.baseline_file(self.case_file)
        return case.directory / self.case_file

    def argv(self) -> list[str]:
        if self.command[0] == "python":
            return [sys.executable, str(paths.BASELINES / self.command[1])]
        return [str(paths.BASELINES / self.command[0])]

    @property
    def directory(self) -> Path:
        """The program's directory under `paths.BASELINES`."""
        return paths.BASELINES / Path(self.command[-1]).parts[0]


_SP1_NOTE = ("SP1's SDK proves from the program and its input, so execution happens inside "
             "the prove call and cannot be excluded")

PROGRAMS: dict[str, Program] = {p.name: p for p in (
    Program("halo2_hw", "handwritten halo2-base (Zinnia's prover stack: IPA/BN254, Blake2b)",
            "target.rs", ("halo2_hw/target/release/halo2_hw",), HALO2_HW_STYLE, (),
            prove_core=("prove_s",), end_to_end=("witness_s", "prove_s"), verify=("verify_s",),
            windows={"prove_core": "create_proof (prove_s)",
                     "end_to_end": "witness synthesis (witness_s) + create_proof (prove_s)",
                     "verify": "verify_proof with the stored verifying key (verify_s)"},
            file_per_function=True),
    Program("sp1", "SP1 6.8.1, core (STARK) proof, Q32 guest", "zkvm.rs",
            ("sp1/target/release/sp1-baseline",), HOST_STYLE, ("--proof-mode", "core"),
            prove_core=("prove",), end_to_end=("prove",), verify=("verify",),
            windows={"prove_core": f"client.prove (prove); {_SP1_NOTE}",
                     "end_to_end": "client.prove (prove), which includes execution",
                     "verify": "client.verify (verify)"}),
    Program("sp1_groth16", "SP1 6.8.1, Groth16-wrapped proof, Q32 guest", "zkvm.rs",
            ("sp1/target/release/sp1-baseline",), HOST_STYLE, ("--proof-mode", "groth16"),
            prove_core=("prove",), end_to_end=("prove",), verify=("verify",),
            windows={"prove_core": f"client.prove in groth16 mode (prove); {_SP1_NOTE}",
                     "end_to_end": "client.prove (prove), which includes execution",
                     "verify": "client.verify (verify)"}),
    Program("risc0", "RISC Zero 3.0.3, composite receipt, Q32 guest", "zkvm.rs",
            ("risc0/target/release/risc0-baseline",), HOST_STYLE, ("--proof-mode", "composite"),
            prove_core=("prove",), end_to_end=("execute", "prove"), verify=("verify",),
            windows={"prove_core": "prove_session on the executed session (prove)",
                     "end_to_end": "executor run (execute) + prove_session (prove)",
                     "verify": "receipt.verify (verify)"}),
    Program("risc0_groth16", "RISC Zero 3.0.3, Groth16 receipt, Q32 guest", "zkvm.rs",
            ("risc0/target/release/risc0-baseline",), HOST_STYLE, ("--proof-mode", "groth16"),
            prove_core=("prove", "compress_succinct", "wrap_groth16"),
            end_to_end=("execute", "prove", "compress_succinct", "wrap_groth16"),
            verify=("verify",),
            windows={"prove_core": "prove_session + succinct compression + Groth16 wrap",
                     "end_to_end": "execute + the prove core",
                     "verify": "receipt.verify (verify)"}),
    Program("cairo", "Cairo (Scarb 2.20.1, Stwo), Q32", "main.cairo",
            ("python", "cairo/cairo_baseline.py"), HOST_STYLE, ("--proof-mode", "stwo"),
            prove_core=("prove",), end_to_end=("execute", "prove"), verify=("verify",),
            windows={"prove_core": "`scarb prove` process wall clock (includes start-up)",
                     "end_to_end": "`scarb execute` + `scarb prove` process wall clocks",
                     "verify": "`scarb verify` process wall clock"}),
)}

KNOWN_ERROR_CLASSES = {"unsupported", "setup_failed", "prove_failed", "witness_dependent",
                       "out_of_memory"}


def copy_tree(source: Path, target: Path) -> None:
    """A copy-on-write copy where the filesystem supports it, a plain copy otherwise."""
    if target.exists():
        shutil.rmtree(target)
    subprocess.run(["cp", "-r", "--reflink=auto", str(source), str(target)], check=True)


def _sum(timings: dict, keys: tuple[str, ...]) -> float | None:
    present = [float(timings[k]) for k in keys if k in timings]
    return sum(present) if present else None


class ContractProgramAdapter(Adapter):
    def __init__(self, program: Program):
        self.program = program
        self.name = program.name
        self.description = program.description
        self.windows = program.windows

    def missing_program(self, case: PerfCase) -> str | None:
        entry = Path(self.program.argv()[-1])
        if not entry.exists():
            return f"no program {paths.relative_to_repo(entry)} (not built?)"
        case_file = self.program.case_file_path(case)
        if not case_file.exists():
            return f"no {case_file.name} in the case folder"
        return None

    def program_files(self, case: PerfCase) -> list[Path]:
        return [self.program.case_file_path(case), Path(self.program.argv()[-1])]

    # -- invoking the program -------------------------------------------------

    def _argv(self, ctx: StepContext, mode: str, input_name: str, work: Path, out: Path,
              extra: list[str]) -> list[str]:
        common = ["--fixture", str(ctx.case.fixture_path), "--input", input_name,
                  "--out", str(out)]
        # A shape-generic folder (NPBench, Pythran) holds one program per entry function.
        function = ["--function", ctx.case.function] if ctx.case.shape_generic else []
        if self.program.style == HALO2_HW_STYLE:
            return [*self.program.argv(), mode, "--case", str(ctx.case.directory), *function,
                    "--workdir", str(work), *common, *self.program.extra_args, *extra]
        return [*self.program.argv(), "--case-dir", str(ctx.case.directory), *function,
                "--mode", mode, "--work-dir", str(work), *common, *self.program.extra_args,
                *extra]

    def _invoke(self, ctx: StepContext, mode: str, input_name: str, work: Path,
                extra: list[str]) -> dict:
        out = work.parent / f"{mode}-{uuid.uuid4().hex[:8]}.json"
        command = self._argv(ctx, mode, input_name, work, out, extra)
        started = time.perf_counter()
        proc = subprocess.run(command, cwd=paths.REPO_ROOT, capture_output=True, text=True)
        wall_s = time.perf_counter() - started
        out.with_suffix(".log").write_text(f"$ {' '.join(command)}\n{proc.stdout}\n{proc.stderr}")
        if proc.returncode == -9:
            raise StepFailure("out_of_memory", f"{self.name} {mode} killed (SIGKILL)")
        if not out.exists():
            raise StepFailure("crashed", f"{self.name} {mode} exited {proc.returncode} "
                              f"without writing {out.name}: {proc.stderr[-500:]}")
        response = json.loads(out.read_text())
        response.setdefault("timings_s", {})["process_wall"] = wall_s
        return response

    @staticmethod
    def _failure(response: dict, default_class: str) -> StepFailure:
        error_class = response.get("error_class")
        failure_class = error_class if error_class in KNOWN_ERROR_CLASSES else default_class
        return StepFailure(failure_class, str(response.get("error", "program reported ok=false")))

    @staticmethod
    def _setup_dir(ctx: StepContext) -> Path:
        return ctx.work_dir / "program"

    @staticmethod
    def _sample_dir(ctx: StepContext) -> Path:
        return ctx.proof_dir / "program"

    # -- the three steps --------------------------------------------------------

    def setup(self, ctx: StepContext) -> SetupResult:
        work = self._setup_dir(ctx)
        work.mkdir(parents=True, exist_ok=True)
        extra: list[str] = []
        override = ctx.options.get("halo2_params")
        if override and self.program.style == HALO2_HW_STYLE:
            extra = ["--k", str(override["k"]), "--lookup-bits", str(override["lookup_bits"])]
        response = self._invoke(ctx, "setup", SETUP_INPUT, work, extra)
        if not response.get("ok"):
            raise self._failure(response, "setup_failed")
        details = {k: v for k, v in response.items() if k not in ("timings_s", "publics", "ok")}
        return SetupResult(timings_s=response["timings_s"], details=details)

    def prove(self, ctx: StepContext) -> ProveResult:
        ctx.proof_dir.mkdir(parents=True, exist_ok=True)
        work = self._sample_dir(ctx)
        copy_tree(self._setup_dir(ctx), work)
        response = self._invoke(ctx, "prove", ctx.input_name, work, [])
        if not response.get("ok"):
            raise self._failure(response, "prove_failed")
        timings = response["timings_s"]
        prove_core = _sum(timings, self.program.prove_core)
        if prove_core is None:
            raise StepFailure("harness_error", f"no {self.program.prove_core} in timings_s")
        proof = Path(response.get("proof_path") or "")
        if not proof.is_file() or not proof.resolve().is_relative_to(work.resolve()):
            raise StepFailure("prove_failed", f"proof {proof} is not a file in {work}")
        shape = response.get("output_shape")
        details = {k: v for k, v in response.items()
                   if k not in ("timings_s", "publics", "ok", "proof_path")}
        return ProveResult(
            prove_core_s=prove_core, end_to_end_s=_sum(timings, self.program.end_to_end),
            proof_path=str(proof), proof_bytes=proof.stat().st_size,
            publics=DecodedPublics(LOGICAL, list(response.get("publics", [])),
                                   shapes=[shape] if shape is not None else None),
            timings_s=timings, details=details)

    def verify(self, ctx: StepContext) -> VerifyResult:
        work = self._sample_dir(ctx)
        extra: list[str] = []
        if ctx.tamper:
            extra = (["--tamper-index", "0"] if self.program.style == HALO2_HW_STYLE
                     else ["--tamper"])
        response = self._invoke(ctx, "verify", ctx.input_name, work, extra)
        timings = response["timings_s"]
        if "verified" not in response:
            error = str(response.get("error", ""))
            if ctx.tamper and any(m in error for m in NO_PUBLIC_VALUE_MESSAGES):
                return VerifyResult(verified=False, verify_s=None, timings_s=timings,
                                    not_applicable="the proof has no public value")
            raise StepFailure("crashed", f"verify reported no verdict: {error[:500]}")
        return VerifyResult(verified=bool(response["verified"]),
                            verify_s=_sum(timings, self.program.verify), timings_s=timings,
                            details={"error": response.get("error")})
