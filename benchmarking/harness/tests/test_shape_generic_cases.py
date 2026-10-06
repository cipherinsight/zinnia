"""NPBench/Pythran cases: shape-generic programs, in-place outputs, shards, groups."""

from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path

import pytest

from benchmarking.harness import paths
from benchmarking.harness.adapters import get_adapter
from benchmarking.harness.analysis.aggregate import summarize
from benchmarking.harness.case_sets import (PerfCase, Tolerance, load_perf_cases, parse_shard,
                                            shard_cases)
from benchmarking.harness.protocol import CaseProtocol, Epoch, RunConfig, StepRunner
from benchmarking.harness.tests.test_protocol import fake_baselines, install_fake  # noqa: F401

IN_PLACE_KERNEL = '''from zinnia import *
import numpy as np


@zk_circuit
def scale(a: Float[Array, "4"]):
    a *= 2.0
'''


def in_place_case(root: Path, name: str = "prog__scale", **fields) -> PerfCase:
    """A shape-generic case whose function writes its argument and returns None.

    The fake programs output `2 * a`, which is what the kernel leaves in `a`.
    """
    case_dir = root / "cases" / name
    case_dir.mkdir(parents=True)
    (case_dir / "kernel.py").write_text(IN_PLACE_KERNEL)
    (case_dir / "zkvm.rs").write_text("// fake program file\n")
    fixture = {"case": name, "function": "scale",
               "parameters": [{"name": "a", "kind": "array", "dtype": "float", "shape": [4]}],
               "inputs": {"A": {"a": [1.0, 2.0, 3.0, 4.0]}, "B": {"a": [0.5, -1.25, 3.0, 8.0]}}}
    (case_dir / "fixture.scale.json").write_text(json.dumps(fixture))
    case = PerfCase(case=name, directory=case_dir, fixture_path=case_dir / "fixture.scale.json",
                    function="scale", tolerance=Tolerance(1e-9, 0.0, "test"), included=True,
                    reason=None, shape_generic=True, mutated_outputs=("a",))
    return replace(case, **fields)


def run_case(root: Path, case: PerfCase, systems: list[str]) -> dict:
    config = RunConfig(systems=systems, repeats=1, step_timeout_s=60.0, memory_max=None)
    protocol = CaseProtocol(StepRunner(Epoch(root), config), config, random.Random(1))
    return {run.adapter.name: run.record for run in protocol.run(case, [get_adapter(s)
                                                                         for s in systems])}


def test_an_in_place_function_outputs_the_arguments_it_writes(tmp_path, fake_baselines):
    install_fake(fake_baselines, "sp1", "honest")
    record = run_case(tmp_path / "epoch", in_place_case(tmp_path), ["sp1"])["sp1"]
    assert record["status"] == "ok"
    assert record["samples"][0]["output_check"]["expected_count"] == 4
    # The host was told which entry function of the folder to run.
    assert record["setup"]["result"]["details"]["function"] == "scale"


def test_a_written_argument_the_case_set_does_not_list_has_no_reference(tmp_path,
                                                                       fake_baselines):
    install_fake(fake_baselines, "sp1", "honest")
    case = in_place_case(tmp_path, mutated_outputs=())
    record = run_case(tmp_path / "epoch", case, ["sp1"])["sp1"]
    assert record["status"] == "reference_failed"
    assert "modifies argument 'a'" in record["failure"]["detail"]


def test_a_fixed_shape_case_gets_no_function_argument(tmp_path, fake_baselines):
    install_fake(fake_baselines, "sp1", "honest")
    case = in_place_case(tmp_path, shape_generic=False)
    record = run_case(tmp_path / "epoch", case, ["sp1"])["sp1"]
    assert record["setup"]["result"]["details"]["function"] is None


def test_case_set_entries_with_a_path_and_entry_function(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "BENCHMARKING", tmp_path)
    case_set = tmp_path / "set.json"
    case_set.write_text(json.dumps({"cases": [
        {"case": "pythran_lu__factorMatrix0", "status": "perf", "path": "datasets/pythran/lu",
         "function": "factorMatrix0", "fixture": "fixture.factorMatrix0.json",
         "shape_generic": True, "mutated_outputs": ["M"], "group": "pythran",
         "float_tolerance": {"atol": 1e-6, "rtol": 1e-6}},
        {"case": "pythran_x__f", "status": "zinnia_rejects", "reason": "rejected",
         "path": "datasets/pythran/x", "function": "f", "group": "pythran"}]}))
    lu, rejected = load_perf_cases(case_set, None, None)
    assert lu.directory == tmp_path / "datasets/pythran/lu"
    assert lu.fixture_path == lu.directory / "fixture.factorMatrix0.json"
    assert (lu.function, lu.shape_generic, lu.mutated_outputs, lu.group) == \
        ("factorMatrix0", True, ("M",), "pythran")
    assert PerfCase.from_json(lu.to_json()) == lu
    assert not rejected.included and rejected.reason == "rejected"
    assert load_perf_cases(case_set, ["pythran_lu__factorMatrix0"], None) == [lu]


def test_shards_are_disjoint_cover_the_set_and_keep_exclusions():
    def case(name: str, included: bool = True) -> PerfCase:
        return PerfCase(case=name, directory=Path(name), fixture_path=Path(name),
                        function="kernel", tolerance=Tolerance(0, 0, "t"), included=included,
                        reason=None if included else "excluded")

    cases = [case(f"c{i}") for i in (5, 1, 4, 2, 3)] + [case("x", included=False)]
    shards = [shard_cases(cases, k, 2) for k in range(2)]
    runs = [[c.case for c in shard if c.included] for shard in shards]
    # Dealt out by sorted id (c1 c3 c5 / c2 c4), listed in the case set's order.
    assert runs == [["c5", "c1", "c3"], ["c4", "c2"]]
    assert all(any(c.case == "x" for c in shard) for shard in shards)
    assert parse_shard("3/8") == (3, 8)
    for bad in ("8/8", "-1/8", "3", "a/b"):
        with pytest.raises(ValueError):
            parse_shard(bad)


def test_the_shards_of_one_run_are_analysed_together_with_groups(tmp_path, fake_baselines):
    install_fake(fake_baselines, "sp1", "honest")
    install_fake(fake_baselines, "risc0", "honest")
    shard_roots = []
    for index, (name, group) in enumerate((("npb__scale", "npbench"),
                                           ("pyt__scale", "pythran"))):
        case = in_place_case(tmp_path / f"cases-{index}", name=name, group=group)
        root = tmp_path / f"shard-{index}"
        run_case(root, case, ["sp1", "risc0"])
        (root / "manifest.json").write_text(json.dumps({"epoch": f"shard-{index}"}))
        (root / "case_set.json").write_text(json.dumps({"cases": [case.to_json()]}))
        shard_roots.append(root)

    summary = summarize(shard_roots, tmp_path / "summary")
    assert summary["coverage"]["sp1"]["ok"] == 2
    assert summary["all_supported_subset"] == ["npb__scale", "pyt__scale"]
    assert summary["comparisons_by_group"]["npbench"]["headline_subset"] == ["npb__scale"]
    assert "## Group pythran" in (tmp_path / "summary" / "summary.md").read_text()

    with pytest.raises(ValueError, match="more than one shard"):
        summarize([shard_roots[0], shard_roots[0]], tmp_path / "twice")
