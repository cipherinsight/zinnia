"""The epoch manifest: what produced an epoch's numbers (audit H-12).

Fields (audit `harness.md` section 4.2): repository state (harness and Zinnia
live in the same repository: commit, `git diff` SHA-256, untracked files), the
extension's build record and current SHA-256, every toolchain version and
path, the halo2_hw `Cargo.lock` hash, machine facts, tracked environment
variables, the run configuration, and the case list. Per-case input-file
hashes are in the raw records.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from benchmarking.harness import build, environment, gitstate, paths, toolchains
from benchmarking.harness.adapters.contract import PROGRAMS
from benchmarking.harness.isolation import memory_capped
from benchmarking.harness.protocol import RunConfig


def _sha256_or_none(path: Path) -> str | None:
    return build.sha256_file(path) if path.is_file() else None


def baseline_programs() -> dict:
    """Program binaries and lock files of the contract programs, by system."""
    out = {}
    for name, program in PROGRAMS.items():
        entry = Path(program.argv()[-1])
        base = program.directory
        out[name] = {
            "program": paths.relative_to_repo(entry),
            "program_sha256": _sha256_or_none(entry),
            "lock_sha256": {paths.relative_to_repo(lock): build.sha256_file(lock)
                            for pattern in ("Cargo.lock", "Scarb.lock")
                            for lock in sorted(base.glob(pattern))} if base.is_dir() else {},
        }
    return out


def start_manifest(epoch_name: str, config: RunConfig, build_record: dict | None,
                   probes: list[toolchains.Probe], case_set: Path | None,
                   case_order: list[str]) -> dict:
    extension = paths.installed_extension()
    child_env = environment.child_environment(config.threads)
    return {
        "schema": 1,
        "epoch": epoch_name,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "repository": gitstate.describe(paths.REPO_ROOT),
        "build": build_record,
        "baseline_builds": (json.loads(build.BASELINE_RECORD.read_text())
                            if build.BASELINE_RECORD.exists() else None),
        "extension_sha256": _sha256_or_none(extension) if extension else None,
        "toolchains": toolchains.as_json(probes),
        "python": toolchains.python_packages(),
        "noir_fpc_sha256": {paths.relative_to_repo(f): build.sha256_file(f)
                            for f in sorted(paths.NOIR_FPC.rglob("*")) if f.is_file()
                            and "target" not in f.parts},
        "baseline_programs": baseline_programs(),
        "machine": environment.machine_info(),
        "cpu_state_at_start": environment.cpu_state(),
        "child_environment": environment.tracked_variables(child_env),
        "parent_environment": environment.tracked_variables(),
        "config": asdict(config),
        "child_command": memory_capped([config.python, "-m", "benchmarking.harness.step",
                                        "REQUEST", "RESPONSE"], config.memory_max),
        "case_set": {"path": paths.relative_to_repo(case_set) if case_set else None,
                     "sha256": _sha256_or_none(case_set) if case_set else None},
        "case_order": case_order,
        "indexer": [],
        "calibration": {},
        "finished_at": None,
    }


def finish_manifest(manifest: dict) -> dict:
    manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest["cpu_state_at_end"] = environment.cpu_state()
    return manifest
