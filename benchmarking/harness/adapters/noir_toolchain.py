"""nargo, Noir's artifact executor and Barretenberg, driven the same way for both Noir sides.

Both Zinnia -> Noir and the handwritten Noir baselines go through this module,
so they use the same binaries (`toolchains.nargo_path()`,
`toolchains.noir_execute_path()`, `toolchains.bb_path()`, never a `PATH`
lookup), the same package layout (the `noir_fpc` library copied into the
package), the same proving scheme and the same flags.

Proofs are zero-knowledge on both sides (author's decision; audit H-08):
UltraHonk with `--verifier_target noir-recursive` (Poseidon2 transcript, ZK).

Witness generation (`prove_sample`) runs setup's compiled artifact with
`noir-execute`, the artifact executor of nargo's Noir release. `nargo execute`
re-elaborates the package source on every call before running the same ACIR,
so it charges the front end to every proof; it still runs in every sample, and
its witness must be identical, so the earlier end-to-end definition stays
computable (`nargo_execute` + `bb_prove`).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from benchmarking.harness import paths, toolchains
from benchmarking.harness.adapters.base import StepFailure, VerifyResult
from benchmarking.harness.outputs import fields_from_be_bytes, fields_to_be_bytes

SCHEME = "ultra_honk"
VERIFIER_TARGET = "noir-recursive"
BB_FLAGS = ("--scheme", SCHEME, "--verifier_target", VERIFIER_TARGET)
#: The end-to-end window of both Noir sides (`prove_sample`).
END_TO_END_WINDOW = ("`noir-execute` on setup's compiled artifact (runs the ACVM: witness "
                     "generation) + `bb prove` with that witness. Every sample also runs "
                     "`nargo execute` (re-elaborates the package source first) after "
                     "`noir-execute` and before `bb prove`, records it as "
                     "`timings_s.nargo_execute` and requires the same witness")
TOOL_TIMEOUT_S = None  # the step timeout of the harness bounds every tool call
OUTPUT_TAIL = 2000


def _tail(proc: subprocess.CompletedProcess) -> str:
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return text[-OUTPUT_TAIL:]


def run_tool(command: list[str], *, cwd: Path, failure_class: str) -> tuple[float, str]:
    """Run one tool; returns (wall seconds, output tail) or raises `StepFailure`."""
    started = time.perf_counter()
    proc = subprocess.run(command, cwd=cwd, capture_output=True, text=True,
                          timeout=TOOL_TIMEOUT_S)
    seconds = time.perf_counter() - started
    if proc.returncode == -9:  # the memory cap's OOM kill lands on the biggest process
        raise StepFailure("out_of_memory", f"{Path(command[0]).name} killed (SIGKILL)")
    if proc.returncode != 0:
        raise StepFailure(failure_class, f"{' '.join(command[:2])} exited "
                          f"{proc.returncode}: {_tail(proc)}")
    return seconds, _tail(proc)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class NoirPackage:
    """A Noir binary package: `Nargo.toml`, `src/main.nr`, vendored `noir_fpc/`."""

    def __init__(self, directory: Path, name: str):
        self.directory = directory
        self.name = name

    @staticmethod
    def create(directory: Path, name: str, main_nr: str,
               libraries: dict[str, Path] | None = None) -> "NoirPackage":
        """A fresh package with `libraries` (name -> source folder) copied in.

        The default library is `noir_fpc`, the only one Zinnia's emitted code uses.
        """
        libraries = {"noir_fpc": paths.NOIR_FPC} if libraries is None else libraries
        if directory.exists():
            shutil.rmtree(directory)
        (directory / "src").mkdir(parents=True)
        (directory / "src" / "main.nr").write_text(main_nr)
        dependencies = "".join(f'{lib} = {{ path = "./{lib}" }}\n' for lib in libraries)
        (directory / "Nargo.toml").write_text(
            f'[package]\nname = "{name}"\ntype = "bin"\nauthors = [""]\n\n'
            f"[dependencies]\n{dependencies}")
        for lib, source in libraries.items():
            shutil.copytree(source, directory / lib, ignore=shutil.ignore_patterns("target"))
        # A copied library that depends on another copied one (noir_hw on noir_fpc)
        # names it by a path relative to its source folder; point it at the copy.
        for lib in libraries:
            manifest = directory / lib / "Nargo.toml"
            text = manifest.read_text()
            for other in libraries:
                text = re.sub(rf'^({other}\s*=\s*\{{\s*path\s*=\s*)"[^"]*"', rf'\1"../{other}"',
                              text, flags=re.M)
            manifest.write_text(text)
        return NoirPackage(directory, name)

    @property
    def target(self) -> Path:
        return self.directory / "target"

    @property
    def bytecode(self) -> Path:
        return self.target / f"{self.name}.json"

    @property
    def vk(self) -> Path:
        return self.target / "vk"

    def write_inputs(self, prover_name: str, toml: str) -> None:
        (self.directory / f"{prover_name}.toml").write_text(toml)

    def compile(self) -> float:
        seconds, _ = run_tool([str(toolchains.nargo_path()), "compile"], cwd=self.directory,
                              failure_class="unsupported")
        return seconds

    def execute(self, prover_name: str, witness_name: str) -> tuple[float, Path]:
        """`nargo execute` with `<prover_name>.toml`: elaborates the package source again,
        checks it against the compiled artifact, then runs the ACIR."""
        seconds, _ = run_tool([str(toolchains.nargo_path()), "execute", "--prover-name",
                               prover_name, witness_name],
                              cwd=self.directory, failure_class="prove_failed")
        return seconds, self.target / f"{witness_name}.gz"

    def execute_artifact(self, prover_name: str, witness_name: str) -> tuple[float, Path]:
        """`noir-execute` with `<prover_name>.toml` on the compiled artifact alone."""
        seconds, _ = run_tool([str(toolchains.noir_execute_path()),
                               "--artifact-path", str(self.bytecode),
                               "--prover-file", str(self.directory / f"{prover_name}.toml"),
                               "--output-dir", str(self.target),
                               "--witness-name", witness_name],
                              cwd=self.directory, failure_class="prove_failed")
        return seconds, self.target / f"{witness_name}.gz"

    def write_vk(self) -> float:
        seconds, _ = run_tool([str(toolchains.bb_path()), "write_vk", "-b", str(self.bytecode),
                               "-o", str(self.target), *BB_FLAGS],
                              cwd=self.directory, failure_class="setup_failed")
        return seconds

    def gates(self) -> dict:
        """`bb gates` for the compiled circuit (circuit size, ACIR opcodes)."""
        _, output = run_tool([str(toolchains.bb_path()), "gates", "-b", str(self.bytecode),
                              *BB_FLAGS], cwd=self.directory, failure_class="setup_failed")
        start = output.find("{")
        try:  # bb prints a status line after the JSON document
            document, _ = json.JSONDecoder().raw_decode(output[start:])
            functions = document.get("functions") or [{}]
        except (ValueError, AttributeError):
            return {"unparsed": output[-300:]}
        first = functions[0]
        return {"circuit_size": first.get("circuit_size"),
                "acir_opcodes": first.get("acir_opcodes")}

    def prove(self, witness: Path, out_dir: Path) -> float:
        from benchmarking.noir_witness import for_barretenberg

        witness = for_barretenberg(self.bytecode, witness)
        out_dir.mkdir(parents=True, exist_ok=True)
        seconds, _ = run_tool([str(toolchains.bb_path()), "prove", "-b", str(self.bytecode),
                               "-w", str(witness), "-k", str(self.vk), "-o", str(out_dir),
                               *BB_FLAGS], cwd=self.directory, failure_class="prove_failed")
        return seconds


def witness_sha256(witness: Path) -> str:
    """SHA-256 of a witness file's decompressed content (the gzip header is not compared)."""
    return hashlib.sha256(gzip.decompress(witness.read_bytes())).hexdigest()


@dataclass
class SampleTimings:
    """The process wall clocks of one Noir prove sample, in seconds."""

    noir_execute: float
    nargo_execute: float
    bb_prove: float
    #: SHA-256 of the decompressed witness both executors wrote.
    witness_sha256: str

    @property
    def end_to_end(self) -> float:
        """Witness generation on the compiled artifact + proving."""
        return self.noir_execute + self.bb_prove

    def timings_s(self) -> dict[str, float]:
        return {"noir_execute": self.noir_execute, "nargo_execute": self.nargo_execute,
                "bb_prove": self.bb_prove}


def prove_sample(package: NoirPackage, prover_name: str, witness_name: str, proof_dir: Path,
                 bytecode_sha256: str) -> SampleTimings:
    """One prove sample on setup's package, in the same order on both Noir sides.

    1. `noir-execute` on the compiled artifact (the timed witness step);
    2. `nargo execute` on the same inputs (recorded as `nargo_execute`, the
       witness step of the earlier end-to-end definition);
    3. the compiled artifact must still be setup's, and the two witnesses
       must be identical once decompressed;
    4. `bb prove` with the witness of step 1.
    """
    execute_s, witness = package.execute_artifact(prover_name, witness_name)
    nargo_s, nargo_witness = package.execute(prover_name, f"{witness_name}_nargo")
    if sha256_file(package.bytecode) != bytecode_sha256:
        raise StepFailure("nondeterministic_build", "nargo execute changed the ACIR")
    digest = witness_sha256(witness)
    nargo_digest = witness_sha256(nargo_witness)
    if digest != nargo_digest:
        raise StepFailure("witness_mismatch", f"noir-execute wrote witness {digest}, nargo "
                          f"execute {nargo_digest} (SHA-256 of the decompressed files)")
    prove_s = package.prove(witness, proof_dir)
    return SampleTimings(noir_execute=execute_s, nargo_execute=nargo_s, bb_prove=prove_s,
                         witness_sha256=digest)


def public_fields(proof_dir: Path) -> list[int]:
    path = proof_dir / "public_inputs"
    return fields_from_be_bytes(path.read_bytes()) if path.exists() else []


def verify_proof(vk: Path, proof_path: Path, tamper: bool, has_outputs: bool) -> VerifyResult:
    """`bb verify` of a proof directory (`proof`, `public_inputs`).

    With `tamper`, the first public value is incremented first (the caller
    passes a copy of the proof directory). `has_outputs=False` means the
    circuit has no public output to tamper with.
    """
    proof_dir = proof_path.parent
    inputs = proof_dir / "public_inputs"
    if tamper:
        fields = public_fields(proof_dir)
        if not fields or not has_outputs:
            return VerifyResult(verified=False, verify_s=None,
                                not_applicable="the proof has no public output")
        fields[0] += 1
        inputs.write_bytes(fields_to_be_bytes(fields))
    command = [str(toolchains.bb_path()), "verify", "-k", str(vk), "-p", str(proof_path),
               "-i", str(inputs), *BB_FLAGS]
    started = time.perf_counter()
    proc = subprocess.run(command, capture_output=True, text=True, timeout=TOOL_TIMEOUT_S)
    seconds = time.perf_counter() - started
    if proc.returncode == -9:
        raise StepFailure("out_of_memory", "bb verify killed (SIGKILL)")
    return VerifyResult(verified=proc.returncode == 0, verify_s=seconds,
                        timings_s={"bb_verify": seconds},
                        details={"exit": proc.returncode,
                                 "output": _tail(proc)[-300:] if proc.returncode else ""})
