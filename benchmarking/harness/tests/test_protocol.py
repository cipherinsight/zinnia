"""The protocol end to end with fake contract programs (no proving system needed).

Each fake program implements the program contract for the toy kernel
`a * 2.0`; misbehaving variants must end in the right failure class.
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import pytest

from benchmarking.harness import paths
from benchmarking.harness.adapters import get_adapter
from benchmarking.harness.analysis.aggregate import summarize
from benchmarking.harness.case_sets import PerfCase, Tolerance
from benchmarking.harness.isolation import memory_capped
from benchmarking.harness.protocol import CaseProtocol, Epoch, RunConfig, StepRunner

FAKE_PROGRAM = Path(__file__).with_name("fake_program.py")
KERNEL = '''from zinnia import *
import numpy as np


@zk_circuit
def kernel(a: Float[Array, "4"]):
    return a * 2.0
'''
PROGRAM_FILES = ("target.rs", "sp1.prog.rs", "risc0.prog.rs", "main.cairo")


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


def install_fake(baselines: Path, directory: str, behavior: str) -> None:
    target = baselines / directory
    target.mkdir(parents=True)
    shutil.copy(FAKE_PROGRAM, target / "bench")
    (target / "bench").chmod(0o755)
    (target / "behavior.txt").write_text(behavior)


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


def test_memory_cap_wraps_in_a_systemd_scope():
    command = memory_capped(["python", "-c", "pass"], "8G")
    assert command[:3] == ["systemd-run", "--user", "--scope"]
    assert "MemoryMax=8G" in command and "MemorySwapMax=0" in command
    assert command[-3:] == ["python", "-c", "pass"]
    assert memory_capped(["true"], None) == ["true"]
