"""`run`: one epoch, i.e. every requested system on every requested case.

Refuses to start when a Zinnia system is requested and the installed extension
is not a recorded release build (audit H-01), when a toolchain the requested
systems need is missing or has the wrong version, or when the memory cap
cannot be applied on this machine.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

from benchmarking.harness import build, environment, isolation, manifest, paths, toolchains
from benchmarking.harness.adapters import ZINNIA_SYSTEMS, get_adapter
from benchmarking.harness.adapters import noir_toolchain
from benchmarking.harness.case_sets import PerfCase
from benchmarking.harness.protocol import CaseProtocol, Epoch, RunConfig, StepRunner

CALIBRATION_REPEATS = 5
CALIBRATION_MAIN_NR = "fn main(x: Field) -> pub Field {\n    x\n}\n"


class RunRefused(RuntimeError):
    """A precondition of a valid measurement does not hold."""


def epoch_directory(label: str, results_dir: Path = paths.RESULTS) -> Path:
    return results_dir / f"{time.strftime('%Y-%m-%d')}-{label}"


def _check_preconditions(config: RunConfig) -> tuple[dict | None, list[toolchains.Probe]]:
    record, problems = build.installed_build_problems()
    if problems and set(config.systems) & set(ZINNIA_SYSTEMS):
        raise RunRefused("the extension cannot be timed: " + "; ".join(problems))
    probes = toolchains.check_environment()
    failed = toolchains.failures_for(probes, config.systems)
    if failed:
        raise RunRefused("toolchain check failed: "
                         + ", ".join(f"{p.name} ({p.status}: {p.found})" for p in failed))
    environment.assert_cpu_only_allowed()
    cap_problem = isolation.cap_problem(config.memory_max)
    if cap_problem:
        raise RunRefused(f"memory cap {config.memory_max}: {cap_problem}")
    return record, probes


def calibrate_bb(work: Path) -> dict:
    """Start-up floor of the Noir CLIs: execute, prove and verify a one-line circuit.

    Both Noir sides pay this floor in every `noir-execute`, `bb prove` and
    `bb verify` window, and in the recorded `nargo execute`. `nargo execute`
    re-elaborates the package's source (and its libraries) before running the
    compiled ACIR, so its floor is mostly nargo's front end. Each repetition
    runs the steps of `noir_toolchain.prove_sample`, in its order.
    """
    package = noir_toolchain.NoirPackage.create(work / "package", "calibration",
                                                CALIBRATION_MAIN_NR)
    package.write_inputs("Prover", 'x = "7"\n')
    package.compile()
    package.write_vk()
    bytecode_sha256 = noir_toolchain.sha256_file(package.bytecode)
    noir_execute_s, nargo_execute_s, prove_s, verify_s = [], [], [], []
    for index in range(CALIBRATION_REPEATS):
        proof_dir = work / f"proof-{index}"
        timings = noir_toolchain.prove_sample(package, "Prover", f"witness-{index}", proof_dir,
                                              bytecode_sha256)
        noir_execute_s.append(timings.noir_execute)
        nargo_execute_s.append(timings.nargo_execute)
        prove_s.append(timings.bb_prove)
        verdict = noir_toolchain.verify_proof(package.vk, proof_dir / "proof", tamper=False,
                                              has_outputs=True)
        verify_s.append(verdict.verify_s)
    return {"circuit": CALIBRATION_MAIN_NR, "noir_execute_s": noir_execute_s,
            "nargo_execute_s": nargo_execute_s, "bb_prove_s": prove_s, "bb_verify_s": verify_s,
            "flags": list(noir_toolchain.BB_FLAGS)}


def run_epoch(label: str, cases: list[PerfCase], config: RunConfig,
              case_set: Path | None, results_dir: Path = paths.RESULTS) -> Path:
    root = epoch_directory(label, results_dir)
    if root.exists():
        raise RunRefused(f"{root} exists; choose another label")
    build_record, probes = _check_preconditions(config)
    rng = random.Random(config.seed)
    included = [c for c in cases if c.included]
    order = list(included)
    rng.shuffle(order)

    root.mkdir(parents=True)
    (root / ".gitignore").write_text("work/\n")
    (root / "case_set.json").write_text(json.dumps(
        {"source": str(case_set) if case_set else None,
         "cases": [c.to_json() for c in cases]}, indent=1))
    record = manifest.start_manifest(root.name, config, build_record, probes, case_set,
                                     [c.case for c in order])
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(record, indent=1))

    epoch = Epoch(root)
    adapters = [get_adapter(name) for name in config.systems]
    protocol = CaseProtocol(StepRunner(epoch, config), config, rng)
    with environment.indexer_paused(record["indexer"]):
        if {"zinnia_noir", "noir_hw"} & set(config.systems):
            record["calibration"]["bb"] = calibrate_bb(root / "work" / "_calibration")
        for position, case in enumerate(order, 1):
            print(f"[{position}/{len(order)}] {case.case}", flush=True)
            for run in protocol.run(case, adapters):
                print(f"    {run.adapter.name:<14} {run.record['status']}", flush=True)
            manifest_path.write_text(json.dumps(record, indent=1))
    manifest_path.write_text(json.dumps(manifest.finish_manifest(record), indent=1))
    print(f"epoch written to {root}")
    return root
