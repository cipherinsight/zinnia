"""zinnia_halo2 through the whole protocol: keys made once in setup, read back per step.

The clocks must cover what halo2_hw's cover (`adapters/zinnia_halo2.py`): no
parameter generation or key generation inside a prove or verify window, the
parameters and the key read from setup's key directory outside the windows,
and the verifier holding only the verifying key.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarking.harness.adapters.zinnia_halo2 import KEYS_DIR, PROOF_FILE, PUBLICS_FILE
from benchmarking.harness.tests.test_protocol import run_protocol


@pytest.fixture(scope="module")
def record(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("zinnia_halo2")
    records, epoch = run_protocol(tmp_path, ["zinnia_halo2"], repeats=2)
    return records["zinnia_halo2"], epoch


def test_the_toy_case_proves_verifies_and_rejects_the_tampered_proof(record):
    rec, _ = record
    assert rec["status"] == "ok", rec["failure"]
    assert rec["tamper"]["status"] == "rejected"
    assert all(s["output_check"]["status"] == "match" for s in rec["samples"])


def test_setup_writes_the_keys_once_without_a_witness(record):
    rec, epoch = record
    setup = rec["setup"]["result"]
    assert set(setup["timings_s"]) == {"compile", "keygen_s", "write_s"}
    assert set(setup["details"]["key_file_bytes"]) == {"keys.json", "params.bin", "pk.bin",
                                                       "vk.bin"}
    key_dir = epoch.work("zinnia_halo2", "toy_case") / KEYS_DIR
    assert sorted(p.name for p in key_dir.iterdir()) == ["keys.json", "params.bin", "pk.bin",
                                                         "vk.bin"]


def test_prove_windows_exclude_key_loading_and_match_halo2_hw(record):
    rec, _ = record
    for sample in rec["samples"]:
        prove = sample["prove"]
        t = prove["timings_s"]
        assert {"load_params_s", "load_pk_s", "parse_inputs_s", "preprocess_s", "witness_s",
                "thread_pool_s", "prove_s", "prove_call_s"} <= set(t)
        assert prove["prove_core_s"] == t["prove_s"]
        assert prove["end_to_end_s"] == pytest.approx(
            t["preprocess_s"] + t["witness_s"] + t["prove_s"])
        assert prove["end_to_end_s"] < t["prove_call_s"]
        assert prove["proof_path"].endswith(PROOF_FILE)


def test_verify_window_holds_only_verify_proof(record):
    rec, _ = record
    for sample in rec["samples"]:
        result = sample["verify"]["result"]
        t = result["timings_s"]
        assert set(t) == {"load_params_s", "load_vk_s", "verify_s", "verify_call_s"}
        assert "load_pk_s" not in t  # the verifier never reads the proving key
        assert result["verify_s"] == t["verify_s"] < t["verify_call_s"]


def test_proof_files_hold_only_the_proof_and_its_publics(record):
    rec, _ = record
    proof_dir = Path(rec["samples"][0]["prove"]["proof_path"]).parent
    assert sorted(p.name for p in proof_dir.iterdir()) == [PROOF_FILE, PUBLICS_FILE]


def test_an_operation_without_a_halo2_base_gadget_is_unsupported(tmp_path):
    """np.tanh has no halo2-base gadget: the program is outside what the backend
    proves (`unsupported`), not a setup that failed."""
    import random

    from benchmarking.harness.adapters import get_adapter
    from benchmarking.harness.protocol import CaseProtocol, Epoch, RunConfig, StepRunner
    from benchmarking.harness.tests.test_protocol import KERNEL, make_case

    case = make_case(tmp_path)
    case.kernel_path.write_text(KERNEL.replace("a * 2.0", "np.tanh(a)"))
    config = RunConfig(systems=["zinnia_halo2"], repeats=1, memory_max=None)
    protocol = CaseProtocol(StepRunner(Epoch(tmp_path / "epoch"), config), config,
                            random.Random(1))
    rec = protocol.run(case, [get_adapter("zinnia_halo2")])[0].record
    assert rec["status"] == "unsupported"
    assert "tanh_f" in rec["failure"]["detail"]
