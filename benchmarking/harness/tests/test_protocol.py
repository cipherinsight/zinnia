"""The protocol end to end with fake contract programs (no proving system needed).

Each fake program implements the program contract for the toy kernel
`a * 2.0`; misbehaving variants must end in the right failure class.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import sys
from pathlib import Path

import pytest

from benchmarking.harness import paths
from benchmarking.harness.adapters import get_adapter
from benchmarking.harness.analysis.aggregate import summarize
from benchmarking.harness.case_sets import PerfCase, Tolerance
from benchmarking.harness import isolation
from benchmarking.harness.isolation import memory_capped
from benchmarking.harness.protocol import CaseProtocol, Epoch, RunConfig, StepRunner

FAKE_PROGRAM = Path(__file__).with_name("fake_program.py")
KERNEL = '''from zinnia import *
import numpy as np


@zk_circuit
def kernel(a: Float[Array, "4"]):
    return a * 2.0
'''
PROGRAM_FILES = ("target.rs", "zkvm.rs", "main.cairo")


def make_case(root: Path) -> PerfCase:
    case_dir = root / "cases" / "toy_case"
    case_dir.mkdir(parents=True)
    (case_dir / "kernel.py").write_text(KERNEL)
    for name in PROGRAM_FILES:
        (case_dir / name).write_text("// fake program file\n")
    fixture = {"case": "toy_case", "function": "kernel",
               "parameters": [{"name": "a", "kind": "array", "dtype": "float", "shape": [4]}],
               "inputs": {"A": {"a": [1.0, 2.0, 3.0, 4.0]}, "B": {"a": [0.5, -1.25, 3.0, 8.0]}}}
    (case_dir / "fixture.json").write_text(json.dumps(fixture))
    return PerfCase(case="toy_case", directory=case_dir, fixture_path=case_dir / "fixture.json",
                    function="kernel", tolerance=Tolerance(1e-9, 0.0, "test"), included=True,
                    reason=None)


def install_fake(baselines: Path, system: str, behavior: str) -> None:
    """The fake program at the path the harness runs for `system`."""
    from benchmarking.harness.adapters.contract import PROGRAMS

    entry = baselines / PROGRAMS[system].command[-1]
    entry.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(FAKE_PROGRAM, entry)
    entry.chmod(0o755)
    (entry.parent / "behavior.txt").write_text(behavior)


@pytest.fixture
def fake_baselines(tmp_path, monkeypatch) -> Path:
    baselines = tmp_path / "baselines"
    monkeypatch.setenv("ZINNIA_HARNESS_BASELINES", str(baselines))
    monkeypatch.setattr(paths, "BASELINES", baselines)
    return baselines


def run_protocol(tmp_path: Path, systems: list[str], timeout_s: float = 60.0,
                 repeats: int = 2) -> tuple[dict, Epoch]:
    case = make_case(tmp_path)
    config = RunConfig(systems=systems, repeats=repeats, step_timeout_s=timeout_s,
                       memory_max=None)
    epoch = Epoch(tmp_path / "epoch")
    protocol = CaseProtocol(StepRunner(epoch, config), config, random.Random(1))
    runs = protocol.run(case, [get_adapter(s) for s in systems])
    return {run.adapter.name: run.record for run in runs}, epoch


def test_failure_classes(tmp_path, fake_baselines):
    install_fake(fake_baselines, "halo2_hw", "honest")
    install_fake(fake_baselines, "sp1", "wrong_output")
    install_fake(fake_baselines, "risc0", "accept_tamper")
    install_fake(fake_baselines, "cairo", "crash_on_prove")
    records, epoch = run_protocol(tmp_path, ["halo2_hw", "sp1", "risc0", "cairo"])

    assert {name: r["status"] for name, r in records.items()} == {
        "halo2_hw": "ok", "sp1": "wrong_output", "risc0": "tamper_accepted", "cairo": "crashed"}
    # Every sample was proved and verified in its own copy of the setup directory.
    proof_dirs = {Path(s["prove"]["proof_path"]).parent for s in records["halo2_hw"]["samples"]}
    assert len(proof_dirs) == 2

    honest = records["halo2_hw"]
    assert [s["tag"] for s in honest["samples"]] == ["sample-1", "sample-2"]
    assert all(s["verify"]["result"]["verified"] for s in honest["samples"])
    assert all(s["output_check"]["status"] == "match" for s in honest["samples"])
    assert honest["tamper"]["status"] == "rejected"
    assert honest["samples"][0]["prove"]["prove_core_s"] == pytest.approx(0.002)
    assert honest["samples"][0]["prove"]["end_to_end_s"] == pytest.approx(0.003)
    # The reversed output is caught position by position (audit M9).
    assert records["sp1"]["failure"]["step"] == "warmup"
    assert records["sp1"]["warmup"]["output_check"]["mismatches"]

    (epoch.root / "manifest.json").write_text(json.dumps({"epoch": "test"}))
    summary = summarize(epoch.root)
    assert summary["coverage"]["halo2_hw"]["ok"] == 1
    assert summary["coverage"]["sp1"]["by_status"] == {"wrong_output": 1}
    assert summary["all_supported_subset"] == []
    assert (epoch.root / "summary" / "per_case.csv").exists()


def test_timeout_is_a_recorded_failure(tmp_path, fake_baselines):
    install_fake(fake_baselines, "halo2_hw", "sleep_on_prove")
    records, _ = run_protocol(tmp_path, ["halo2_hw"], timeout_s=5.0, repeats=1)
    assert records["halo2_hw"]["status"] == "timeout"
    assert records["halo2_hw"]["failure"]["step"] == "warmup"


def test_missing_program_is_no_program(tmp_path, fake_baselines):
    install_fake(fake_baselines, "halo2_hw", "honest")
    records, _ = run_protocol(tmp_path, ["halo2_hw", "cairo"], repeats=1)
    assert records["halo2_hw"]["status"] == "ok"
    assert records["cairo"]["status"] == "no_program"


def test_memory_cap_wraps_in_a_systemd_scope(monkeypatch):
    monkeypatch.delenv(isolation.CAP_VARIABLE, raising=False)
    command = memory_capped(["python", "-c", "pass"], "8G")
    assert command[:3] == ["systemd-run", "--user", "--scope"]
    assert "MemoryMax=8G" in command and "MemorySwapMax=0" in command
    assert "OOMPolicy=continue" in command
    assert command[-3:] == ["python", "-c", "pass"]
    assert memory_capped(["true"], None) == ["true"]


def test_a_tool_killed_at_the_memory_cap_is_out_of_memory(tmp_path, fake_baselines,
                                                         monkeypatch):
    """The fake program allocates past a 300M cap in a grandchild during prove."""
    monkeypatch.setenv(isolation.CAP_VARIABLE, "systemd")
    if isolation.cap_problem("300M"):
        pytest.skip("needs a working systemd user manager")
    install_fake(fake_baselines, "halo2_hw", "allocate_on_prove")
    case = make_case(tmp_path)
    config = RunConfig(systems=["halo2_hw"], repeats=1, step_timeout_s=60.0, memory_max="300M")
    protocol = CaseProtocol(StepRunner(Epoch(tmp_path / "epoch"), config), config,
                            random.Random(1))
    record = protocol.run(case, [get_adapter("halo2_hw")])[0].record
    assert record["status"] == "out_of_memory"
    assert record["failure"]["step"] == "warmup"


def test_watchdog_mechanism_starts_the_command_without_systemd(monkeypatch):
    monkeypatch.setenv(isolation.CAP_VARIABLE, "watchdog")
    assert memory_capped(["true"], "8G") == ["choom", "-n", "1000", "--", "true"]
    assert isolation.cap_problem("8G") is None
    assert isolation.cap_problem("eight gigabytes") is not None
    monkeypatch.setenv(isolation.CAP_VARIABLE, "cgroups")
    with pytest.raises(ValueError):
        isolation.cap_mechanism()


def test_memory_sizes_use_systemd_units():
    assert isolation.size_in_bytes("16G") == 16 << 30
    assert isolation.size_in_bytes("300M") == 300 << 20
    assert isolation.size_in_bytes("1.5k") == 1536
    assert isolation.size_in_bytes("4096") == 4096


HOLD_MEMORY = ("import time; x = bytearray(400 << 20); x[::4096] = b'1' * len(x[::4096]); "
               "time.sleep(60)")


def test_watchdog_kills_a_session_over_the_cap(tmp_path, monkeypatch):
    """A grandchild holding 400 MB under a 200M cap: the whole session dies quickly."""
    monkeypatch.setenv(isolation.CAP_VARIABLE, "watchdog")
    command = [sys.executable, "-c",
               f"import subprocess, sys; subprocess.run([sys.executable, '-c', {HOLD_MEMORY!r}])"]
    outcome = isolation.run_child(command, env=dict(os.environ), cwd=tmp_path,
                                  timeout_s=30.0, memory_max="200M", log_path=tmp_path / "log")
    assert outcome.over_memory_cap and outcome.killed_by_sigkill and not outcome.timed_out
    assert outcome.peak_rss_bytes > 200 << 20
    assert outcome.wall_s < 20
    assert "over the 200M cap" in (tmp_path / "log").read_text()


def test_watchdog_leaves_a_small_step_alone(tmp_path, monkeypatch):
    monkeypatch.setenv(isolation.CAP_VARIABLE, "watchdog")
    outcome = isolation.run_child([sys.executable, "-c", "import time; time.sleep(0.5)"],
                                  env=dict(os.environ), cwd=tmp_path,
                                  timeout_s=30.0, memory_max="1G", log_path=tmp_path / "log")
    assert outcome.returncode == 0 and not outcome.over_memory_cap
    assert 0 < outcome.peak_rss_bytes < 1 << 30
