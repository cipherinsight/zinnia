"""The staged handwritten Noir library retains its relative dependencies."""
import json
from types import SimpleNamespace
from pathlib import Path

from benchmarking import noir_gates


def test_handwritten_library_stages_transitive_dependency(monkeypatch):
    calls = []

    def run(command, *, cwd=None, **kwargs):
        calls.append(command)
        root = Path(cwd) if cwd else Path(command[3]).parents[1]
        assert (root / "baselines/noir_hw/Nargo.toml").is_file()
        assert (root / "baselines/noir_hw/../../noir_fpc/Nargo.toml").resolve().is_file()
        assert 'path = "./baselines/noir_hw"' in (root / "Nargo.toml").read_text()
        return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps({
            "functions": [{"circuit_size": 42, "acir_opcodes": 3}]}))

    monkeypatch.setattr(noir_gates.subprocess, "run", run)
    assert noir_gates.gate_count("fn main() {}", "noir_hw", noir_gates.NOIR_HW) == {
        "gates": 42, "acir_opcodes": 3}
    assert [c[1] for c in calls] == ["compile", "gates"]
