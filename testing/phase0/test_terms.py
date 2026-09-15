"""Unit tests for the term evaluator (`terms.py`).

`terms.py` must agree with `src/phase0/term/mod.rs` node for node, because the
adequacy verdict is only as good as the evaluation of `dims` and `defcond`.
Two kinds of case:

* hand-built DAGs, which pin down floor `div`/`mod`, `wrap64` as `term/mod.rs`
  composes it, `min`/`max`/`ite` and the Boolean connectives;
* the export of a real circuit, whose `*_pretty` fields are what the Rust
  printer produced, checked against `TermDag.pretty`.

```sh
.venv/bin/python -m pytest testing/phase0/test_terms.py -q
```
"""

from __future__ import annotations

from typing import Any

import pytest

from testing.phase0.fixtures import binary, dag_of, integer  # noqa: E402
from testing.phase0.terms import TermDag, UnboundVariable  # noqa: E402

#: `2^63` and `2^64`, the constants `term/mod.rs::wrap64` composes with.
TWO_POW_63 = 1 << 63
TWO_POW_64 = 1 << 64


def test_floor_division_matches_python_and_the_rust() -> None:
    """`div` and `mod` are floor semantics for a negative operand too
    (`term/mod.rs::floor_div`).
    """
    dag = dag_of(
        [
            {"id": 0, "sort": "int", "op": "var", "var": 0},
            {"id": 1, "sort": "int", "op": "div", "k": "3", "args": [0]},
            {"id": 2, "sort": "int", "op": "mod", "k": "3", "args": [0]},
            {"id": 3, "sort": "int", "op": "div", "k": "-3", "args": [0]},
        ],
        [("v", "Idx")],
    )
    for value in (-7, -3, -1, 0, 1, 7):
        theta = {0: value}
        assert dag.evaluate(1, theta) == value // 3
        assert dag.evaluate(2, theta) == value % 3
        assert dag.evaluate(3, theta) == value // -3


def test_wrap64_as_term_rs_composes_it() -> None:
    """`wrap64(t) = ((t + 2^63) mod 2^64) - 2^63`, built from `add`, `mod` and
    `sub` — there is no `wrap64` node (`term/mod.rs::wrap64`).
    """
    dag = dag_of(
        [
            {"id": 0, "sort": "int", "op": "var", "var": 0},
            integer(1, TWO_POW_63),
            binary(2, "add", [0, 1]),
            {"id": 3, "sort": "int", "op": "mod", "k": str(TWO_POW_64), "args": [2]},
            binary(4, "sub", [3, 1]),
        ],
        [("v", "I64")],
    )
    assert dag.evaluate(4, {0: 0}) == 0
    assert dag.evaluate(4, {0: TWO_POW_63 - 1}) == TWO_POW_63 - 1
    assert dag.evaluate(4, {0: TWO_POW_63}) == -TWO_POW_63
    assert dag.evaluate(4, {0: TWO_POW_63 + 5}) == -TWO_POW_63 + 5


def test_min_max_ite_and_the_connectives() -> None:
    dag = dag_of(
        [
            {"id": 0, "sort": "int", "op": "var", "var": 0},
            integer(1, 4),
            binary(2, "min", [0, 1]),
            binary(3, "max", [0, 1]),
            binary(4, "lt", [0, 1], sort="bool"),
            binary(5, "ite", [4, 2, 3]),
            {"id": 6, "sort": "bool", "op": "not", "args": [4]},
            binary(7, "or", [4, 6], sort="bool"),
            binary(8, "and", [4, 6], sort="bool"),
        ],
        [("v", "Idx")],
    )
    assert dag.evaluate(5, {0: 1}) == 1  # v < 4, so min(v, 4)
    assert dag.evaluate(5, {0: 9}) == 9  # otherwise max(v, 4)
    assert dag.evaluate_bool(7, {0: 1}) is True
    assert dag.evaluate_bool(8, {0: 1}) is False


def test_a_boolean_variable_reads_the_assignment() -> None:
    dag = dag_of(
        [
            {"id": 0, "sort": "bool", "op": "bvar", "var": 0},
            integer(1, 10),
            integer(2, 20),
            binary(3, "ite", [0, 1, 2]),
        ],
        [("b", "Bool")],
    )
    assert dag.evaluate(3, {0: True}) == 10
    assert dag.evaluate(3, {0: False}) == 20


def test_an_unbound_variable_is_an_error_not_a_default() -> None:
    dag = dag_of([{"id": 0, "sort": "int", "op": "var", "var": 0}], [("v", "Idx")])
    with pytest.raises(UnboundVariable):
        dag.evaluate(0, {})


def test_closure_is_sorted_and_closed_under_children() -> None:
    dag = dag_of(
        [integer(0, 1), integer(1, 2), binary(2, "add", [0, 1]),
         integer(3, 99), binary(4, "add", [2, 0])],
        [],
    )
    assert dag.closure([4]) == [0, 1, 2, 4]
    assert dag.variables(4) == set()


# ---------------------------------------------------------------------------
# Against a real export
# ---------------------------------------------------------------------------


def _analysis() -> dict[str, Any]:
    """Phase 0's record for a small circuit with a mask, a slice and a merge."""
    from zinnia import Array, Integer, NDArray, ZKCircuit, zk_circuit

    @zk_circuit
    def sample(x: NDArray[Integer, 8], m: NDArray[Integer, 8], k: Integer):
        s = x[m][:4]
        total = s.sum()
        if total > 0:
            out = total * 2
        else:
            out = total + k
        return out

    del Array
    return ZKCircuit.from_method(sample).analyze_shapes()


def test_pretty_matches_the_rust_printer() -> None:
    """Every `*_pretty` field of a real export is what `TermDag.pretty`
    produces, which is the cheapest check that the two DAG readers agree.
    """
    analysis = _analysis()
    assert "rejected" not in analysis, analysis.get("rejected")
    dag = TermDag.from_json(analysis["terms"])
    checked = 0
    for constraint in analysis["constraints"]:
        assert dag.pretty(int(constraint["formula"])) == constraint["pretty"]
        checked += 1
    for entry in analysis["descriptors"]:
        checked += _check_descriptor_pretty(dag, entry["desc"])
    assert checked > 0, "the sample circuit produced no terms to compare"


def _check_descriptor_pretty(dag: TermDag, desc: dict[str, Any]) -> int:
    if desc["kind"] == "composite":
        return sum(_check_descriptor_pretty(dag, e) for e in desc.get("elems") or [])
    checked = 0
    for term, pretty in zip(desc.get("dims") or [], desc.get("dims_pretty") or []):
        assert dag.pretty(int(term)) == pretty
        checked += 1
    if desc.get("term") is not None:
        assert dag.pretty(int(desc["term"]["id"])) == desc["term_pretty"]
        checked += 1
    return checked
