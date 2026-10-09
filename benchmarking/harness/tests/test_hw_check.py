"""The handwritten-baseline checker must not silently accept invalid outputs or coverage."""
import json
import sys

import pytest

from benchmarking.baselines.hw_check import check
from benchmarking.baselines.hw_check.entries import Entry
from benchmarking.baselines.hw_check.summarize import summarize


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True])
def test_non_numeric_or_nonfinite_float_output_is_rejected(value):
    assert check.compare([value], [("float", 1.0)], 1e-6, 1e-6)


@pytest.mark.parametrize("value", [2, -1, "true", 1.0])
def test_boolean_output_must_be_a_bit(value):
    assert check.compare([value], [("bool", True)], 1e-6, 1e-6)


def test_valid_outputs_still_match():
    assert not check.compare([1.0000001, 1, False, 42],
                             [("float", 1.0), ("bool", True), ("bool", False), ("int", 42)],
                             1e-6, 1e-6)


def test_missing_requested_baseline_fails(tmp_path, monkeypatch):
    fixture = tmp_path / "fixture.json"
    fixture.write_text("{}")
    entry = Entry("example__kernel", tmp_path, "kernel", fixture, status="perf")
    (tmp_path / "target.kernel.rs").write_text("// present")
    out = tmp_path / "results.json"
    monkeypatch.setattr(check, "npb_entries", lambda: {entry.case: entry})
    monkeypatch.setattr(check.CaseCheck, "__init__", lambda *args, **kwargs: None)
    monkeypatch.setattr(check.CaseCheck, "halo2", lambda self: {"verdict": "pass"})
    monkeypatch.setattr(sys, "argv", ["check", "--npb", "--perf", "--out", str(out)])
    assert check.main() == 1
    result = json.loads(out.read_text())[entry.case]
    assert result["halo2"]["verdict"] == "pass"
    assert result["noir"]["verdict"] == "FAIL"
    assert "missing baseline" in result["noir"]["problems"][0]


def test_size_summary_keeps_missing_entries_and_uses_geometric_mean():
    rows = [{"halo2": {"ratio": r}} for r in (0.25, 1.0, 4.0, None)]
    result = summarize(rows, "halo2")
    assert result["entries"] == 4
    assert result["compared"] == 3
    assert result["geomean_zinnia_over_handwritten"] == pytest.approx(1.0)
    assert result["zinnia_smaller"] == result["equal"] == result["zinnia_larger"] == 1
