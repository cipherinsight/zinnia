"""Hand-written records in the export's own JSON shape.

`acceptance.md` §3.1 makes every comparison a pure function of two JSON
records, so a unit test can hand it one it built itself. These are the
builders that produce them — a term arena (:class:`Arena`), one descriptor
entry, one baseline record — and they are shared by `test_consistency_units.py`
and `test_terms.py` so that a new test need not copy either.

The shapes are the ones `src/phase0/export.rs` documents: `nodes[i].id == i`,
every argument a smaller id, and every integer constant a decimal *string*.
"""

from __future__ import annotations

from typing import Any

from testing.phase0.baseline import BaselineRecord
from testing.phase0.terms import TermDag


class Arena:
    """A term arena built by hand.

    It appends nodes in the order the export guarantees (`nodes[i].id == i`,
    every argument a smaller id), so the records it builds are the shape
    `consistency/` and `adequacy.py` read.
    """

    def __init__(self) -> None:
        self.vars: list[dict[str, Any]] = []
        self.nodes: list[dict[str, Any]] = []
        self.variables: list[dict[str, Any]] = []

    def _node(self, payload: dict[str, Any]) -> int:
        payload["id"] = len(self.nodes)
        self.nodes.append(payload)
        return payload["id"]

    def const(self, value: int) -> int:
        return self._node({"sort": "int", "op": "const", "value": str(value)})

    def true(self) -> int:
        return self._node({"sort": "bool", "op": "true", "args": []})

    def var(self, name: str, lo: int, hi: int) -> int:
        var_id = len(self.vars)
        self.vars.append({"id": var_id, "name": name, "sort": "Idx"})
        node = self._node({"sort": "int", "op": "var", "var": var_id})
        self.variables.append(
            {
                "id": var_id,
                "name": name,
                "sort": "Idx",
                "lo": self.const(lo),
                "lo_pretty": str(lo),
                "hi": self.const(hi),
                "hi_pretty": str(hi),
                "origin": {"kind": "input_length"},
                "element_path": [],
            }
        )
        return node

    def binary(self, op: str, left: int, right: int, sort: str = "int") -> int:
        return self._node({"sort": sort, "op": op, "args": [left, right]})

    def div(self, child: int, k: int) -> int:
        return self._node({"sort": "int", "op": "div", "k": str(k), "args": [child]})

    def record(self, **overrides: Any) -> dict[str, Any]:
        """A phase 0 record carrying this arena, with empty defaults."""
        payload = {
            "schema": "zinnia.phase0/6",
            "config": {"loop_limit": 1000, "recursion_limit": 16,
                       "work_budget": 4_000_000},
            "terms": {"vars": self.vars, "nodes": self.nodes},
            "variables": self.variables,
            "constraints": [],
            "products": [],
            "descriptors": [],
            "delivery": [],
            "smtlib": "",
        }
        payload.update(overrides)
        return payload


def array_descriptor(name: str, dims: list[int], psi: int) -> dict[str, Any]:
    """One `descriptors` entry holding a rank-`len(dims)` array.

    Its producer is `input`, the one kind that names no other value, so the
    entry stands on its own (`implementation.md` §5a).
    """
    return {
        "key": {"key": name, "stmt_path": [0], "iters": [], "calls": [],
                "expr_index": 0, "step": 0},
        "name": name,
        "checks": 0,
        "producer": {"kind": "input", "alias": None, "rules": None,
                     "operands": None, "condition": None},
        "desc": {
            "kind": "array",
            "dtype": "Float",
            "rank": len(dims),
            "dims": dims,
            "dims_pretty": ["?"] * len(dims),
            "defcond": psi,
            "psi": psi,
        },
    }


def baseline_with(**overrides: Any) -> BaselineRecord:
    record = BaselineRecord(program="p", function="f", status="compiled")
    for name, value in overrides.items():
        setattr(record, name, value)
    return record


def dag_of(nodes: list[dict[str, Any]], variables: list[tuple[str, str]]) -> TermDag:
    """A DAG in the export's own JSON shape, from raw node dicts.

    `Arena` is the builder to reach for when the test only needs *a* DAG;
    this one is for the tests that pin down a specific node layout
    (`test_terms.py`).
    """
    return TermDag.from_json({
        "vars": [{"id": i, "name": n, "sort": s} for i, (n, s) in enumerate(variables)],
        "nodes": nodes,
    })


def integer(node_id: int, value: int) -> dict[str, Any]:
    """One `const` node. The value is a decimal string, as the export writes
    it, so that an `i128` outside the JSON-safe range survives exactly.
    """
    return {"id": node_id, "sort": "int", "op": "const", "value": str(value)}


def binary(node_id: int, op: str, args: list[int], sort: str = "int") -> dict[str, Any]:
    """One node with evaluated arguments and no constant of its own."""
    return {"id": node_id, "sort": sort, "op": op, "args": args}
