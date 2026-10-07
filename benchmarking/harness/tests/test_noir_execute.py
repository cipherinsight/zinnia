"""Noir witness generation: the pinned `noir-execute`, the prove sample both Noir
sides run, and the two end-to-end clocks of the analysis."""

from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path

import pytest

from benchmarking.harness import build, paths, toolchains
from benchmarking.harness.adapters import noir_toolchain
from benchmarking.harness.adapters.base import StepFailure
from benchmarking.harness.analysis import aggregate

COMMIT = "b4236c1957d0c26cb65d82adc9e5447b6ff1d629"
NARGO_VERSION_OUTPUT = (f"nargo version = 1.0.0-beta.20\n"
                        f"noirc version = 1.0.0-beta.20+{COMMIT}\n"
                        f"(git version hash: {COMMIT}, is dirty: false)\n")


# ---------------------------------------------------------------------------
# The commit check
# ---------------------------------------------------------------------------


def fake_nargo(tmp_path: Path, output: str) -> Path:
    script = tmp_path / "nargo"
    script.write_text(f"#!/bin/sh\ncat <<'EOF'\n{output}EOF\n")
    script.chmod(0o755)
    return script


def test_nargo_commit_is_read_from_its_version(tmp_path, monkeypatch):
    monkeypatch.setattr(toolchains, "nargo_path", lambda: fake_nargo(tmp_path,
                                                                     NARGO_VERSION_OUTPUT))
    assert toolchains.nargo_commit() == COMMIT


def test_a_dirty_nargo_has_no_commit(tmp_path, monkeypatch):
    dirty = NARGO_VERSION_OUTPUT.replace("is dirty: false", "is dirty: true")
    monkeypatch.setattr(toolchains, "nargo_path", lambda: fake_nargo(tmp_path, dirty))
    with pytest.raises(RuntimeError, match="modified checkout"):
        toolchains.nargo_commit()


@pytest.fixture
def recorded_build(tmp_path, monkeypatch):
    """A fake installed `noir-execute` and its build record, for nargo at `COMMIT`."""
    binary = tmp_path / "bin" / "noir-execute"
    binary.parent.mkdir()
    binary.write_bytes(b"binary")
    record_path = tmp_path / "record.json"
    record = {"noir_commit": COMMIT, "binary_sha256": build.sha256_file(binary),
              "source": {"dirty": False}}
    record_path.write_text(json.dumps(record))
    monkeypatch.setattr(paths, "NOIR_EXECUTE_BINARY", binary)
    monkeypatch.setattr(paths, "NOIR_EXECUTE_RECORD", record_path)
    monkeypatch.setattr(toolchains, "nargo_commit", lambda: COMMIT)
    return binary, record_path, record


def test_the_recorded_build_of_nargos_commit_passes(recorded_build):
    probe = toolchains.probe_noir_execute_build()
    assert (probe.status, probe.expected) == ("ok", COMMIT)
    assert toolchains.failures_for([probe], ["noir_hw"]) == []


def test_another_commit_is_refused(recorded_build, monkeypatch):
    monkeypatch.setattr(toolchains, "nargo_commit", lambda: "0" * 40)
    probe = toolchains.probe_noir_execute_build()
    assert probe.status == "mismatch" and f"built from {COMMIT}" in probe.found
    assert toolchains.failures_for([probe], ["zinnia_noir"]) == [probe]
    assert toolchains.failures_for([probe], ["zinnia_halo2"]) == []


def test_a_replaced_binary_is_refused(recorded_build):
    binary, _, _ = recorded_build
    binary.write_bytes(b"another binary")
    probe = toolchains.probe_noir_execute_build()
    assert probe.status == "mismatch" and "not the recorded build" in probe.found


def test_a_dirty_source_is_refused(recorded_build):
    _, record_path, record = recorded_build
    record_path.write_text(json.dumps({**record, "source": {"dirty": True}}))
    assert toolchains.probe_noir_execute_build().status == "mismatch"


def test_no_build_record_is_a_failure(recorded_build):
    _, record_path, _ = recorded_build
    record_path.unlink()
    probe = toolchains.probe_noir_execute_build()
    assert probe.status == "missing" and "build --skip-zinnia --noir-execute" in probe.found


# ---------------------------------------------------------------------------
# One prove sample
# ---------------------------------------------------------------------------


class ScriptedPackage(noir_toolchain.NoirPackage):
    """A package whose tools write given witnesses and log the order of the calls."""

    def __init__(self, directory: Path, artifact_witness: bytes, nargo_witness: bytes):
        super().__init__(directory, "toy")
        self.target.mkdir(parents=True)
        self.bytecode.write_text("{}")
        self.witnesses = {"artifact": artifact_witness, "nargo": nargo_witness}
        self.calls: list[str] = []

    def _write(self, kind: str, witness_name: str) -> tuple[float, Path]:
        self.calls.append(kind)
        path = self.target / f"{witness_name}.gz"
        path.write_bytes(gzip.compress(self.witnesses[kind], mtime=len(self.calls)))
        return {"artifact": 0.1, "nargo": 0.7}[kind], path

    def execute_artifact(self, prover_name, witness_name):
        return self._write("artifact", witness_name)

    def execute(self, prover_name, witness_name):
        return self._write("nargo", witness_name)

    def prove(self, witness, out_dir):
        self.calls.append(f"prove {witness.name}")
        return 2.0


def test_a_sample_proves_the_artifact_executors_witness(tmp_path):
    package = ScriptedPackage(tmp_path / "pkg", b"witness", b"witness")
    timings = noir_toolchain.prove_sample(package, "Prover_B", "w", tmp_path / "proof",
                                          noir_toolchain.sha256_file(package.bytecode))
    # Same order on both Noir sides; the gzip headers differ, the content does not.
    assert package.calls == ["artifact", "nargo", "prove w.gz"]
    assert timings.timings_s() == {"noir_execute": 0.1, "nargo_execute": 0.7, "bb_prove": 2.0}
    assert timings.end_to_end == pytest.approx(2.1)
    assert timings.witness_sha256 == noir_toolchain.witness_sha256(package.target / "w_nargo.gz")


def test_different_witnesses_fail_before_proving(tmp_path):
    package = ScriptedPackage(tmp_path / "pkg", b"witness", b"other witness")
    with pytest.raises(StepFailure) as failure:
        noir_toolchain.prove_sample(package, "Prover_B", "w", tmp_path / "proof",
                                    noir_toolchain.sha256_file(package.bytecode))
    assert failure.value.failure_class == "witness_mismatch"
    assert package.calls == ["artifact", "nargo"]


def test_a_changed_artifact_fails_before_proving(tmp_path):
    package = ScriptedPackage(tmp_path / "pkg", b"witness", b"witness")
    with pytest.raises(StepFailure) as failure:
        noir_toolchain.prove_sample(package, "Prover_B", "w", tmp_path / "proof", "0" * 64)
    assert failure.value.failure_class == "nondeterministic_build"
    assert "prove w.gz" not in package.calls


@pytest.mark.skipif(not all(p.is_file() for p in (toolchains.nargo_path(), toolchains.bb_path(),
                                                   toolchains.noir_execute_path())),
                    reason="nargo, bb or noir-execute not installed")
def test_a_sample_with_the_real_tools(tmp_path):
    main_nr = ("fn main(x: Field, y: [Field; 3]) -> pub Field {\n"
               "    let mut acc = x;\n    for i in 0..3 { acc = acc * y[i] + 1; }\n    acc\n}\n")
    package = noir_toolchain.NoirPackage.create(tmp_path / "pkg", "toy", main_nr)
    package.write_inputs("Prover_B", 'x = "3"\ny = ["2", "5", "7"]\n')
    package.compile()
    package.write_vk()
    timings = noir_toolchain.prove_sample(package, "Prover_B", "w", tmp_path / "proof",
                                          noir_toolchain.sha256_file(package.bytecode))
    assert timings.witness_sha256 == noir_toolchain.witness_sha256(package.target / "w.gz")
    assert noir_toolchain.public_fields(tmp_path / "proof") == [((3 * 2 + 1) * 5 + 1) * 7 + 1]
    honest = noir_toolchain.verify_proof(package.vk, tmp_path / "proof" / "proof",
                                         tamper=False, has_outputs=True)
    assert honest.verified
    shutil.copytree(tmp_path / "proof", tmp_path / "tamper")
    tampered = noir_toolchain.verify_proof(package.vk, tmp_path / "tamper" / "proof",
                                           tamper=True, has_outputs=True)
    assert not tampered.verified


# ---------------------------------------------------------------------------
# The two end-to-end clocks
# ---------------------------------------------------------------------------


def record(system: str, *proves: dict) -> dict:
    return {"system": system, "samples": [{"prove": prove} for prove in proves]}


def test_end_to_end_with_the_artifact_executor_and_with_nargo():
    noir = record("zinnia_noir",
                  {"end_to_end_s": 2.1, "timings_s": {"noir_execute": 0.1,
                                                      "nargo_execute": 0.7, "bb_prove": 2.0}})
    assert aggregate.clock_values(noir, "end_to_end_s") == [2.1]
    assert aggregate.clock_values(noir, "end_to_end_nargo_s") == [pytest.approx(2.7)]
    assert aggregate.end_to_end_witness_step(noir) == "noir-execute"


def test_an_earlier_noir_epoch_measured_end_to_end_with_nargo():
    earlier = record("noir_hw", {"end_to_end_s": 0.7 + 2.0,
                                 "timings_s": {"nargo_execute": 0.7, "bb_prove": 2.0}})
    assert aggregate.clock_values(earlier, "end_to_end_nargo_s") == \
        aggregate.clock_values(earlier, "end_to_end_s")
    assert aggregate.end_to_end_witness_step(earlier) == "nargo execute"


def test_other_systems_have_one_end_to_end_clock():
    halo2 = record("zinnia_halo2", {"end_to_end_s": 1.5, "timings_s": {"prove_s": 1.0}})
    assert aggregate.clock_values(halo2, "end_to_end_nargo_s") == [1.5]
    assert aggregate.end_to_end_witness_step(halo2) is None
    assert "end_to_end_nargo_s" not in aggregate.clocks_of("zinnia_halo2", "halo2_hw")
    assert "end_to_end_nargo_s" in aggregate.clocks_of("zinnia_noir", "sp1")
