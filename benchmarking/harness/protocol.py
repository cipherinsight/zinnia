"""The measurement protocol (audit `harness.md` section 4.1) and the failure taxonomy.

For each case (cases in a seeded random order) and each requested system:

1. setup with input A (compile, keys), in its own child process;
2. one warm-up prove with input B, discarded from the statistics;
3. `repeats` timed proves with input B, one child process each, the systems
   interleaved: sample i runs every system before sample i+1 starts, in an
   order rotated per sample, so slow drift (thermal, background load) spreads
   over all systems;
4. every timed proof verified in its own child process;
5. every proof's public outputs compared, in order, with the NumPy reference
   on input B (`outputs.check_outputs`);
6. once per case, a copy of the first proof with one public value changed must
   be rejected by the verifier.

Every step runs under the same memory cap and the same timeout. A failure ends
that system's work on the case and is kept in the raw record with its class
(`FAILURE_CLASSES`); nothing is dropped. A raw record's status is `ok` only if
setup, every prove and verify succeeded, every output matched, and the tampered
proof was rejected (or had no public value to tamper with).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from benchmarking.harness import environment, paths
from benchmarking.harness.adapters.base import Adapter, sha256_path
from benchmarking.harness.case_sets import PerfCase
from benchmarking.harness.isolation import run_child
from benchmarking.harness.outputs import DecodedPublics, check_outputs, leaves_from_json

OK = "ok"

FAILURE_CLASSES: dict[str, str] = {
    "no_program": "the system has no program for this case",
    "reference_failed": "the kernel body does not run as plain NumPy on input B",
    "unsupported": "the system's compiler rejected the program",
    "setup_failed": "key generation or another setup step failed",
    "witness_dependent": "the circuit built for input A differs from the one for input B",
    "nondeterministic_build": "a later step compiled a different circuit than setup did",
    "prove_failed": "witness generation or proving failed on input B",
    "witness_mismatch": "Noir's artifact executor and `nargo execute` wrote different witnesses",
    "verify_failed": "the verifier rejected an honest proof",
    "wrong_output": "the proof's public outputs differ from the NumPy reference",
    "output_unchecked": "the public outputs could not be mapped onto the kernel's outputs",
    "tamper_accepted": "the verifier accepted a proof with one public value changed",
    "tamper_inconclusive": "the tamper check did not produce a verdict",
    "timeout": "a step exceeded the per-step timeout",
    "out_of_memory": "a step was killed at the memory cap",
    "crashed": "a step's process died without writing a result",
    "harness_error": "the harness itself failed",
}

REFERENCE_SYSTEM = "_reference"


@dataclass
class RunConfig:
    systems: list[str]
    repeats: int = 5
    step_timeout_s: float = 1800.0
    memory_max: str | None = "16G"
    threads: int | None = None
    seed: int = 20261006
    halo2_params: dict | None = None
    #: [K, N] when the epoch runs shard K of N of the case set (`case_sets.shard_cases`).
    shard: list[int] | None = None
    python: str = field(default_factory=lambda: sys.executable)

    def options(self) -> dict:
        return {"halo2_params": self.halo2_params} if self.halo2_params else {}


@dataclass
class Epoch:
    root: Path

    def raw(self, system: str, case: str) -> Path:
        return self.root / "raw" / system / f"{case}.json"

    def work(self, system: str, case: str) -> Path:
        return self.root / "work" / system / case

    def logs(self, system: str, case: str) -> Path:
        return self.root / "logs" / system / case


# ---------------------------------------------------------------------------
# Running one step in a child process
# ---------------------------------------------------------------------------


class StepRunner:
    """Writes a step request, runs `benchmarking.harness.step` capped, reads the response."""

    def __init__(self, epoch: Epoch, config: RunConfig):
        self.epoch = epoch
        self.config = config
        self.env = environment.child_environment(config.threads)

    def run(self, system: str, step: str, case: PerfCase, input_name: str, tag: str,
            **extra: Any) -> dict:
        work_dir = self.epoch.work(system, case.case)
        steps_dir = work_dir / "steps"
        steps_dir.mkdir(parents=True, exist_ok=True)
        request = {"system": system, "step": step, "case": case.to_json(), "input": input_name,
                   "work_dir": str(work_dir), "options": self.config.options(),
                   **{k: (str(v) if isinstance(v, Path) else v) for k, v in extra.items()}}
        request_path = steps_dir / f"{tag}.request.json"
        response_path = steps_dir / f"{tag}.response.json"
        request_path.write_text(json.dumps(request))
        response_path.unlink(missing_ok=True)
        outcome = run_child(
            [self.config.python, "-m", "benchmarking.harness.step", str(request_path),
             str(response_path)],
            env=self.env, cwd=paths.REPO_ROOT, timeout_s=self.config.step_timeout_s,
            memory_max=self.config.memory_max,
            log_path=self.epoch.logs(system, case.case) / f"{tag}.log")
        if response_path.exists() and not outcome.timed_out:
            response = json.loads(response_path.read_text())
        else:
            response = {"ok": False, "result": None,
                        "failure": {"class": _dead_child_class(outcome),
                                    "detail": f"exit {outcome.returncode} after "
                                              f"{outcome.wall_s:.1f}s"}}
        response["child"] = {"wall_s": outcome.wall_s, "returncode": outcome.returncode,
                             "timed_out": outcome.timed_out,
                             "peak_rss_bytes": outcome.peak_rss_bytes,
                             "over_memory_cap": outcome.over_memory_cap}
        return response


def _dead_child_class(outcome) -> str:
    if outcome.timed_out:
        return "timeout"
    if outcome.over_memory_cap or outcome.killed_by_sigkill:
        return "out_of_memory"
    return "crashed"


# ---------------------------------------------------------------------------
# One system on one case
# ---------------------------------------------------------------------------


@dataclass
class SystemRun:
    """The raw record of one system on one case, built step by step."""

    adapter: Adapter
    case: PerfCase
    record: dict
    #: (proof directory, proof file) of every timed sample, in order.
    proofs: list[tuple[Path, Path]] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return self.record["failure"] is not None

    def fail(self, failure_class: str, step: str, detail: str) -> None:
        if self.record["failure"] is None:
            self.record["failure"] = {"class": failure_class, "step": step, "detail": detail}

    def fail_from(self, response: dict, step: str) -> None:
        failure = response["failure"]
        self.fail(failure["class"], step, failure["detail"])


def _file_hashes(files: list[Path]) -> dict[str, str | None]:
    return {paths.relative_to_repo(f): sha256_path(f) for f in files}


def new_system_run(adapter: Adapter, case: PerfCase, snapshot: dict) -> SystemRun:
    files = [case.kernel_path, case.fixture_path, *adapter.program_files(case)]
    record = {
        "schema": 1, "system": adapter.name, "case": case.case,
        "description": adapter.description, "windows": adapter.windows,
        "status": None, "failure": None, "files": _file_hashes(files),
        "tolerance": asdict(case.tolerance), "environment": snapshot,
        "setup": None, "warmup": None, "samples": [], "tamper": None,
    }
    return SystemRun(adapter, case, record)


def _publics_digest(publics: dict) -> str:
    return hashlib.sha256(json.dumps(publics, sort_keys=True).encode()).hexdigest()


class CaseProtocol:
    """Runs every requested system on one case."""

    def __init__(self, runner: StepRunner, config: RunConfig, rng: random.Random):
        self.runner = runner
        self.config = config
        self.rng = rng

    def run(self, case: PerfCase, adapters: list[Adapter]) -> list[SystemRun]:
        snapshot = environment.cpu_state()
        runs = [new_system_run(adapter, case, snapshot) for adapter in adapters]
        reference = self._reference(case)
        if not reference["ok"]:
            for run in runs:
                run.fail_from(reference, "reference")
            return self._finish(runs)
        expected = (leaves_from_json(reference["result"]["leaves"]),
                    reference["result"]["shapes"])

        active = []
        for run in runs:
            missing = run.adapter.missing_program(case)
            if missing:
                run.fail("no_program", "availability", missing)
            else:
                active.append(run)
        self.rng.shuffle(active)

        for run in active:
            self._setup(run)
        for run in self._live(active, 0):
            self._prove(run, "warmup", expected)
        for index in range(1, self.config.repeats + 1):
            for run in self._live(active, index):
                self._prove(run, f"sample-{index}", expected)
        for run in active:
            self._verify_all(run)
            self._tamper(run)
        return self._finish(runs)

    def _live(self, active: list[SystemRun], rotation: int) -> list[SystemRun]:
        """Systems still running, in the case order rotated by `rotation`."""
        if not active:
            return []
        shift = rotation % len(active)
        rotated = active[shift:] + active[:shift]
        return [r for r in rotated if not r.failed]

    def _reference(self, case: PerfCase) -> dict:
        response = self.runner.run(REFERENCE_SYSTEM, "reference", case, "B", "reference")
        record = {"case": case.case, **response,
                  "files": _file_hashes([case.kernel_path, case.fixture_path])}
        path = self.runner.epoch.raw(REFERENCE_SYSTEM, case.case)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record))
        return response

    def _setup(self, run: SystemRun) -> None:
        response = self.runner.run(run.adapter.name, "setup", run.case, "A", "setup")
        run.record["setup"] = response
        if not response["ok"]:
            run.fail_from(response, "setup")

    def _prove(self, run: SystemRun, tag: str, expected) -> None:
        if run.failed:
            return
        proof_dir = self.runner.epoch.work(run.adapter.name, run.case.case) / "proofs" / tag
        if proof_dir.exists():
            shutil.rmtree(proof_dir)
        load = list(os.getloadavg())
        response = self.runner.run(run.adapter.name, "prove", run.case, "B", tag,
                                   proof_dir=proof_dir)
        sample: dict[str, Any] = {"tag": tag, "loadavg_before": load,
                                  "child": response["child"]}
        if tag == "warmup":
            run.record["warmup"] = sample
        else:
            run.record["samples"].append(sample)
        if not response["ok"]:
            sample["failure"] = response["failure"]
            run.fail_from(response, tag)
            return
        result = response["result"]
        publics = result.pop("publics")
        leaves, shapes = expected
        check = check_outputs(leaves, DecodedPublics.from_json(publics), run.case.tolerance,
                              expected_shapes=shapes)
        sample.update(prove=result, output_check=check.to_json(),
                      publics_sha256=_publics_digest(publics))
        if tag in ("warmup", "sample-1"):
            sample["publics"] = publics
        if tag != "warmup":
            run.proofs.append((proof_dir, Path(result["proof_path"])))
        if check.status == "mismatch":
            run.fail("wrong_output", tag, check.detail)

    def _verify_all(self, run: SystemRun) -> None:
        for sample, (proof_dir, proof_path) in zip(run.record["samples"], run.proofs):
            if run.failed:
                return
            tag = f"verify-{sample['tag']}"
            response = self.runner.run(run.adapter.name, "verify", run.case, "B", tag,
                                       proof_dir=proof_dir, proof_path=proof_path)
            sample["verify"] = response
            if not response["ok"]:
                run.fail_from(response, tag)
            elif not response["result"]["verified"]:
                run.fail("verify_failed", tag,
                         json.dumps(response["result"].get("details"))[:2000])

    def _tamper(self, run: SystemRun) -> None:
        """Verify a copy of the first sample with one public value changed."""
        if run.failed or not run.proofs:
            return
        proof_dir, proof_path = run.proofs[0]
        copy_dir = proof_dir.parent / "tamper"
        if copy_dir.exists():
            shutil.rmtree(copy_dir)
        shutil.copytree(proof_dir, copy_dir, symlinks=True)
        response = self.runner.run(run.adapter.name, "verify", run.case, "B", "tamper",
                                   proof_dir=copy_dir,
                                   proof_path=copy_dir / proof_path.relative_to(proof_dir),
                                   tamper=True)
        run.record["tamper"] = {"response": response}
        if not response["ok"]:
            run.record["tamper"]["status"] = "error"
            run.fail("tamper_inconclusive", "tamper", response["failure"]["detail"][:2000])
            return
        result = response["result"]
        if result.get("not_applicable"):
            run.record["tamper"]["status"] = "not_applicable"
        elif result["verified"]:
            run.record["tamper"]["status"] = "accepted"
            run.fail("tamper_accepted", "tamper", "a changed public value verified")
        else:
            run.record["tamper"]["status"] = "rejected"

    def _finish(self, runs: list[SystemRun]) -> list[SystemRun]:
        for run in runs:
            if not run.failed:
                unchecked = [s for s in run.record["samples"]
                             if s.get("output_check", {}).get("status") == "unchecked"]
                if unchecked:
                    run.fail("output_unchecked", unchecked[0]["tag"],
                             unchecked[0]["output_check"]["detail"])
            run.record["status"] = OK if not run.failed else run.record["failure"]["class"]
            path = self.runner.epoch.raw(run.adapter.name, run.case.case)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(run.record, indent=1))
        return runs
