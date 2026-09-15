"""Unit tests for the five comparisons of `consistency/`, on hand-written
records (`revision/phase0/acceptance.md` §3.1).

Every comparison is a pure function of phase 0's JSON record and the
baseline's, so none of these needs a corpus, a built extension or a cached
compile. `test_consistency.py` is the driver over the parity set; this is
the part that can run in a tenth of a second.

```sh
.venv/bin/python -m pytest testing/phase0/test_consistency_units.py -q
```
"""

from __future__ import annotations

from typing import Any

import z3

from testing.phase0 import consistency as C
from testing.phase0.baseline import (
    BaselineInput,
    DynamicAllocation,
    ReturnExposure,
    StaticAllocation,
)
from testing.phase0.fixtures import Arena, array_descriptor, baseline_with
from testing.phase0.phi_solver import PhiSolver


def test_static_match_pairs_by_size_in_allocation_order() -> None:
    arena = Arena()
    truth = arena.true()
    small, large = arena.const(4), arena.const(9)
    record = arena.record(
        descriptors=[
            array_descriptor("big", [large], truth),
            array_descriptor("small", [small], truth),
        ]
    )
    dag = C.TermDag.from_json(record["terms"])
    entries = C.allocated_arrays(record, dag)
    baseline = baseline_with(
        static_allocations=[
            StaticAllocation(stmt_id=7, segment_id=1, size=4, init_value=0),
            StaticAllocation(stmt_id=2, segment_id=0, size=9, init_value=0),
        ]
    )
    matches = C.match_static_arrays(entries, baseline)
    assert [(m.size, m.verdict) for m in matches] == [(9, "matched"), (4, "matched")]
    assert matches[0].phase0.startswith("big")


def test_static_match_reports_both_sides() -> None:
    arena = Arena()
    truth = arena.true()
    record = arena.record(descriptors=[array_descriptor("a", [arena.const(3)], truth)])
    dag = C.TermDag.from_json(record["terms"])
    entries = C.allocated_arrays(record, dag)
    baseline = baseline_with(
        static_allocations=[StaticAllocation(stmt_id=0, segment_id=0, size=5, init_value=0)]
    )
    verdicts = [m.verdict for m in C.match_static_arrays(entries, baseline)]
    assert verdicts == ["unmatched_baseline", "unmatched_phase0"]


def _bounded_record() -> tuple[dict[str, Any], C.TermDag]:
    """One array whose single extent is `n + 1`, with `0 <= n <= 7`."""
    arena = Arena()
    n = arena.var("n", 0, 7)
    dims = arena.binary("add", n, arena.const(1))
    truth = arena.true()
    record = arena.record(
        constraints=[
            {
                "formula": arena.binary(
                    "and",
                    arena.binary("le", arena.const(0), n, sort="bool"),
                    arena.binary("le", n, arena.const(7), sort="bool"),
                    sort="bool",
                ),
                "pretty": "0 <= n <= 7",
                "kind": "var_range",
                "provenance": "range of n",
                "var": 0,
            }
        ],
        descriptors=[array_descriptor("a", [dims], truth)],
    )
    return record, C.TermDag.from_json(record["terms"])


def test_dynamic_bound_is_tighter_than_a_larger_envelope() -> None:
    record, dag = _bounded_record()
    baseline = baseline_with(
        dynamic_allocations=[
            DynamicAllocation(0, 0, "float", 64, 1, [64], param_name=None)
        ]
    )
    solver = PhiSolver(record, budget_seconds=30.0)
    entries = C.allocated_arrays(record, dag)
    matches = C.compare_dynamic_envelopes(entries, baseline, dag, solver)
    axis = matches[0].axes[0]
    assert (matches[0].match, axis.verdict, axis.phase0, axis.baseline) == (
        "order", "tighter", 8, 64,
    )


def test_dynamic_bound_is_looser_than_a_smaller_envelope() -> None:
    record, dag = _bounded_record()
    baseline = baseline_with(
        dynamic_allocations=[
            DynamicAllocation(0, 0, "float", 4, 1, [4], param_name=None)
        ]
    )
    solver = PhiSolver(record, budget_seconds=30.0)
    entries = C.allocated_arrays(record, dag)
    matches = C.compare_dynamic_envelopes(entries, baseline, dag, solver)
    assert matches[0].axes[0].verdict == "looser"
    assert matches[0].fails


def test_dynamic_match_prefers_the_witnessed_parameter() -> None:
    record, dag = _bounded_record()
    baseline = baseline_with(
        dynamic_allocations=[
            DynamicAllocation(9, 1, "float", 64, 1, [64], param_name=None),
            DynamicAllocation(0, 0, "float", 8, 1, [8], param_name="a"),
        ]
    )
    solver = PhiSolver(record, budget_seconds=30.0)
    entries = C.allocated_arrays(record, dag)
    matches = C.compare_dynamic_envelopes(entries, baseline, dag, solver)
    by_match = {m.match: m for m in matches}
    assert by_match["param"].phase0 == "a"
    assert by_match["param"].axes[0].verdict == "equal"
    assert by_match["unmatched_baseline"].baseline.startswith("dyn array 1")


def test_input_shape_mismatch_is_reported() -> None:
    arena = Arena()
    truth = arena.true()
    three = arena.const(3)
    record = arena.record(
        descriptors=[
            {
                "key": {
                    "key": "|||0", "stmt_path": [], "iters": [], "calls": [],
                    "expr_index": 0,
                },
                "name": "x",
                "desc": {
                    "kind": "array", "dtype": "Float", "rank": 1,
                    "dims": [three], "dims_pretty": ["3"],
                    "defcond": truth, "psi": truth,
                },
            }
        ]
    )
    dag = C.TermDag.from_json(record["terms"])
    baseline = baseline_with(
        inputs=[
            BaselineInput(
                name="x", kind="Private",
                declared={"NDArray": {"shape": [4], "dtype": "Float"}},
                resolved={"NDArray": {"shape": [4], "dtype": "Float"}},
            )
        ]
    )
    check = C.compare_inputs(record, baseline, dag, solver=None)[0]
    assert check.verdict == "shape_differs" and check.fails


def test_floor_division_follows_python_not_smtlib() -> None:
    """`term/mod.rs` divides with floor semantics; SMT-LIB's `div` is Euclidean.

    `-7 div -2` is `3` in Python and `4` under SMT-LIB's rule, so the rewrite
    of `print.rs::smt_floor_div` is what keeps the harness honest.
    """
    arena = Arena()
    quotient = arena.div(arena.const(-7), -2)
    record = arena.record()
    solver = PhiSolver(record, budget_seconds=5.0)
    model = z3.Solver()
    model.add(solver.term(quotient) == -7 // -2)
    assert model.check() == z3.sat


def _delivered_float_array(extent: int) -> tuple[list[C.ArrayEntry], PhiSolver]:
    """A circuit whose delivery set is one Float array of `extent` cells."""
    arena = Arena()
    truth = arena.true()
    entry = array_descriptor("return", [arena.const(extent)], truth)
    entry["key"]["key"] = "|||4294967295"
    record = arena.record(descriptors=[entry], delivery=[{"key": "|||4294967295"}])
    dag = C.TermDag.from_json(record["terms"])
    return C.descriptor_leaves(record, dag, delivery=True), PhiSolver(record, 5.0)


def test_exposure_split_drives_the_payload_check() -> None:
    delivered, solver = _delivered_float_array(6)
    exposure = ReturnExposure(public_int=0, public_float=4, elided_int=0, elided_float=2)
    check = C.check_return_payload(delivered, baseline_with(exposure=exposure), solver)
    assert (check.verdict, check.bounds) == ("equal", [6])


def test_payload_check_stands_down_when_the_baseline_binds_integers() -> None:
    """An integer cell may be payload of another dtype, or shape metadata."""
    delivered, solver = _delivered_float_array(6)
    exposure = ReturnExposure(public_int=1, public_float=0, elided_int=0, elided_float=0)
    check = C.check_return_payload(delivered, baseline_with(exposure=exposure), solver)
    assert check.verdict == "not_applicable" and "dtype" in check.detail


def test_payload_check_stands_down_on_an_empty_output() -> None:
    """A circuit the baseline compiled to no output states no envelope."""
    delivered, solver = _delivered_float_array(6)
    exposure = ReturnExposure(public_int=0, public_float=0, elided_int=0, elided_float=0)
    check = C.check_return_payload(delivered, baseline_with(exposure=exposure), solver)
    assert check.verdict == "not_applicable" and "no return cell" in check.detail
