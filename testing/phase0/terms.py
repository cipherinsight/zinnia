"""The term evaluator of `revision/phase0/acceptance.md` §2.3.

Phase 0 exports its term DAG as `{"vars": [...], "nodes": [...]}` (the schema
is in `src/phase0/export.rs`, the DAG in `src/phase0/term/mod.rs` and its
printers in `src/phase0/term/print.rs`). This module rebuilds that DAG in
Python and evaluates any node under an assignment $\\theta(u)$, mirroring
`TermArena::eval` node for node:

* integer nodes are unbounded mathematical integers, as in the Rust `i128`;
* `div` and `mod` are floor semantics, which is exactly Python's `//` and `%`;
* `wrap64` is not a node — `term/mod.rs::wrap64` builds it from `add`, `mod`
  and `sub`, so it needs no case here;
* `min`, `max`, `ite` and the Boolean connectives are the obvious ones.

Nodes are topologically ordered by construction (`nodes[i].id == i` and every
argument is a smaller id), so evaluation is one upward sweep with no
recursion, like the Rust.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

class UnboundVariable(KeyError):
    """A term mentions a variable the assignment does not bind."""

    def __init__(self, var_id: int, name: str) -> None:
        super().__init__(f"variable {var_id} (`{name}`) has no value in theta(u)")
        self.var_id = var_id
        self.name = name


@dataclass(frozen=True)
class VarDecl:
    """One row of `terms.vars`: the printer's name and the sort."""

    id: int
    name: str
    sort: str  # "Idx" | "I64" | "Bool"

    @property
    def is_bool(self) -> bool:
        return self.sort == "Bool"


@dataclass
class TermDag:
    """The exported term DAG, ready to evaluate."""

    nodes: list[dict[str, Any]]
    vars: list[VarDecl] = field(default_factory=list)

    @staticmethod
    def from_json(terms: dict[str, Any]) -> "TermDag":
        """Build the DAG from the `terms` object of the phase 0 JSON."""
        declarations = [
            VarDecl(id=int(v["id"]), name=v["name"], sort=v["sort"])
            for v in terms["vars"]
        ]
        return TermDag(nodes=list(terms["nodes"]), vars=declarations)

    def var(self, var_id: int) -> VarDecl:
        return self.vars[var_id]

    # ── evaluation ────────────────────────────────────────────────────

    def evaluate(self, node_id: int, theta: dict[int, Any]) -> Any:
        """The value of node `node_id` under `theta` (var id -> int | bool).

        Raises `UnboundVariable` when a reachable variable has no value.
        """
        values = self._evaluate_all(node_id, theta)
        return values[node_id]

    def evaluate_bool(self, node_id: int, theta: dict[int, Any]) -> bool:
        """The value of a Boolean node, as a `bool`."""
        return bool(self.evaluate(node_id, theta))

    def closure(self, roots: Iterable[int]) -> list[int]:
        """The nodes `roots` depend on, sorted ascending.

        The adequacy harness evaluates the same set of roots — every `dims`,
        `defcond`, $\\psi$ and constraint of one circuit — once per draw, so
        the closure is computed once and the sweep below reuses it.
        """
        roots = list(roots)
        if not roots:
            return []
        needed = set(roots)
        for index in range(max(roots), -1, -1):
            if index in needed:
                needed.update(self._children(self.nodes[index]))
        return sorted(needed)

    def valuate(self, order: list[int], theta: dict[int, Any]) -> dict[int, Any]:
        """Evaluate every node of `order` in one upward sweep.

        `order` must be ascending and closed under children, which is what
        `closure` returns.
        """
        values: dict[int, Any] = {}
        for index in order:
            values[index] = self._evaluate_node(self.nodes[index], values, theta)
        return values

    def _evaluate_all(self, root: int, theta: dict[int, Any]) -> dict[int, Any]:
        return self.valuate(self.closure([root]), theta)

    @staticmethod
    def _children(node: dict[str, Any]) -> Iterable[int]:
        return [int(a) for a in node.get("args", ())]

    def _evaluate_node(
        self, node: dict[str, Any], values: dict[int, Any], theta: dict[int, Any]
    ) -> Any:
        """One node, given its arguments' values.

        The four operators that read a field of the node itself — `const`'s
        `value`, `var`/`bvar`'s `var`, and the constant `k` of `mul`, `div`
        and `mod` — are handled here; everything else is a function of the
        argument values alone and lives in :data:`_OPERATORS`, so that the
        table reads like `_PRETTY_FORMS` and like `phi_solver._APPLY` and the
        three can be diffed against `term/print.rs` line by line.
        """
        op = node["op"]
        if op == "const":
            return int(node["value"])
        if op in ("var", "bvar"):
            return self._variable(int(node["var"]), theta, boolean=op == "bvar")
        args = [values[int(a)] for a in node.get("args", ())]
        if op in ("mul", "div", "mod"):
            return _SCALED[op](args[0], int(node["k"]))
        apply = _OPERATORS.get(op)
        if apply is None:
            raise ValueError(f"unknown term operator `{op}` in node {node.get('id')}")
        return apply(args)

    def _variable(self, var_id: int, theta: dict[int, Any], boolean: bool) -> Any:
        if var_id not in theta:
            raise UnboundVariable(var_id, self.vars[var_id].name)
        value = theta[var_id]
        return bool(value) if boolean else int(value)

    def variables(self, node_id: int) -> set[int]:
        """Every variable of $\\Phi$ the term mentions."""
        return {
            int(self.nodes[index]["var"])
            for index in self.closure([node_id])
            if self.nodes[index]["op"] in ("var", "bvar")
        }

    # ── printing ──────────────────────────────────────────────────────

    def pretty(self, node_id: int) -> str:
        """The rendering `TermArena::pretty` produces, for cross-checking the
        `*_pretty` fields of the export.
        """
        rendered: dict[int, str] = {}
        for index in self.closure([node_id]):
            rendered[index] = self._pretty_node(self.nodes[index], rendered)
        return rendered[node_id]

    def _pretty_node(self, node: dict[str, Any], rendered: dict[int, str]) -> str:
        op = node["op"]
        args = [rendered[int(a)] for a in node.get("args", ())]
        if op == "const":
            return node["value"]
        if op in ("var", "bvar"):
            return self.vars[int(node["var"])].name
        if op == "mul":
            return f"({node['k']} * {args[0]})"
        if op in ("div", "mod"):
            return f"({args[0]} {op} {node['k']})"
        return _PRETTY_FORMS[op].format(*args)


#: The operators whose constant `k` lives on the node (`term/print.rs`'s
#: `mul`, `div` and `mod`). Division is floor, which is Python's `//`.
_SCALED = {
    "mul": lambda arg, k: k * arg,
    "div": lambda arg, k: arg // k,
    "mod": lambda arg, k: arg % k,
}

#: Every other operator, as a function of the evaluated arguments. One row
#: per node kind of `term/print.rs::node_json`.
_OPERATORS = {
    "true": lambda args: True,
    "false": lambda args: False,
    "add": lambda args: args[0] + args[1],
    "sub": lambda args: args[0] - args[1],
    "min": lambda args: min(args[0], args[1]),
    "max": lambda args: max(args[0], args[1]),
    "ite": lambda args: args[1] if args[0] else args[2],
    "eq": lambda args: args[0] == args[1],
    "lt": lambda args: args[0] < args[1],
    "le": lambda args: args[0] <= args[1],
    "not": lambda args: not args[0],
    "and": lambda args: bool(args[0]) and bool(args[1]),
    "or": lambda args: bool(args[0]) or bool(args[1]),
}

#: The remaining `pretty` forms, keyed by operator (see `term/print.rs::render_node`).
_PRETTY_FORMS = {
    "add": "({} + {})",
    "sub": "({} - {})",
    "min": "min({}, {})",
    "max": "max({}, {})",
    "ite": "ite({}, {}, {})",
    "true": "true",
    "false": "false",
    "eq": "({} = {})",
    "lt": "({} < {})",
    "le": "({} <= {})",
    "not": "!{}",
    "and": "({} && {})",
    "or": "({} || {})",
}
