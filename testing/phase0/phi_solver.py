"""$\\Phi$ as a Z3 problem: the solver layer of `acceptance.md` §3.

`consistency/` asks three questions of $\\Phi$ — is this bound at most the
baseline's, is this formula entailed, is this term constant — and this module
is where they are answered. It holds two pieces:

* :class:`Z3Arena`, one Z3 expression per node of the exported term DAG,
  built in a single upward sweep. `testing/phase0/terms.py` evaluates the same
  DAG under an assignment; this evaluates it symbolically, node for node,
  with the floor `div` / `mod` rewrites `src/phase0/term/print.rs` states.
* :class:`PhiSolver`, the constraints of $\\Phi$ as reusable solvers, with
  the per-circuit time budget every query is charged against.

The three decisions behind the queries — the DAG rather than the SMT-LIB
text, CANFAIL answered without the check it is about, and the product facts
of §3.4 asserted and dropped on `unknown` — are argued in `consistency/`'s
module docstring, which is where a reader meets them.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import z3

from testing.phase0.terms import TermDag

#: Milliseconds one Z3 query may take before it answers `unknown`.
QUERY_TIMEOUT_MS = int(os.environ.get("ZINNIA_PHASE0_Z3_TIMEOUT_MS", "3000"))

#: Wall-clock seconds of Z3 work per circuit, across every query.
CIRCUIT_BUDGET_SECONDS = float(os.environ.get("ZINNIA_PHASE0_Z3_BUDGET", "180"))

#: Term-arena size past which the Z3 stage is skipped for a circuit. Building
#: one Z3 expression per node is linear, but a 10^6-node arena of an unrolled
#: nest costs more memory than the answer is worth.
MAX_ARENA_NODES = int(os.environ.get("ZINNIA_PHASE0_MAX_ARENA", "400000"))


@dataclass(frozen=True)
class Answer:
    """What Z3 said, for every query of this module.

    `verdict` is the query's own vocabulary — `within` / `looser` /
    `unknown` for :meth:`PhiSolver.exceeds`, `bounded` / `unbounded` /
    `unsatisfiable` / `unknown` for :meth:`PhiSolver.maximize_expr`,
    `entailed` / `refutable` / `vacuous` / `unknown` for
    :meth:`PhiSolver.entails`, `constant <n>` / `not_constant` / `vacuous` /
    `unknown` for :meth:`PhiSolver.is_constant`, and the four verdicts of
    :meth:`PhiSolver.bound`. `value` is the maximum where the query found
    one, `seconds` what it cost, and `note` why the answer is weaker than it
    could be (a dropped product fact, a timeout).

    One type for all five, so that a caller unpacks by name and cannot get
    the order of a tuple wrong.
    """

    verdict: str
    value: int | None = None
    seconds: float = 0.0
    note: str | None = None


# ---------------------------------------------------------------------------
# The term DAG in Z3
# ---------------------------------------------------------------------------


class Z3Arena:
    """One Z3 expression per node of a phase 0 term arena.

    Built in a single upward sweep, which the export's ordering allows
    (`nodes[i].id == i` and every argument is a smaller id). `terms.py`
    evaluates the same DAG under an assignment; this evaluates it under Z3's
    symbolic algebra, node for node, with the floor semantics of
    `src/phase0/term/print.rs`.
    """

    def __init__(self, dag: TermDag) -> None:
        self.dag = dag
        self.variables = {
            decl.id: (z3.Bool(decl.name) if decl.is_bool else z3.Int(decl.name))
            for decl in dag.vars
        }
        self.nodes: list[Any] = []
        for node in dag.nodes:
            self.nodes.append(self._build(node))

    def __getitem__(self, node_id: int) -> Any:
        return self.nodes[int(node_id)]

    def _build(self, node: dict[str, Any]) -> Any:
        op = node["op"]
        args = [self.nodes[int(a)] for a in node.get("args", ())]
        if op == "const":
            return z3.IntVal(int(node["value"]))
        if op in ("var", "bvar"):
            return self.variables[int(node["var"])]
        if op == "mul":
            return z3.IntVal(int(node["k"])) * args[0]
        if op == "div":
            return _floor_div(args[0], int(node["k"]))
        if op == "mod":
            return _floor_mod(args[0], int(node["k"]))
        if op == "true":
            return z3.BoolVal(True)
        if op == "false":
            return z3.BoolVal(False)
        return _APPLY[op](*args)


def _floor_div(child: Any, k: int) -> Any:
    """Python's floor division in Z3, whose integer `/` is Euclidean.

    The rewrite is `src/phase0/term/print.rs::smt_floor_div`: for `k > 0` the
    two agree; for `k < 0`, `floor(a/k) = floor(-a/-k)`, whose divisor is
    positive. `k` is a non-zero constant by the grammar of §3.1.
    """
    return child / z3.IntVal(k) if k > 0 else (-child) / z3.IntVal(-k)


def _floor_mod(child: Any, k: int) -> Any:
    """Python's `%` in Z3; see `print.rs::smt_floor_mod`."""
    if k > 0:
        return child % z3.IntVal(k)
    return child + z3.IntVal(-k) * ((-child) / z3.IntVal(-k))


#: The operators whose arguments are already built, by name.
_APPLY = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "min": lambda a, b: z3.If(a <= b, a, b),
    "max": lambda a, b: z3.If(a <= b, b, a),
    "ite": z3.If,
    "eq": lambda a, b: a == b,
    "lt": lambda a, b: a < b,
    "le": lambda a, b: a <= b,
    "not": z3.Not,
    "and": z3.And,
    "or": z3.Or,
}


# ---------------------------------------------------------------------------
# Φ as a solver
# ---------------------------------------------------------------------------


class PhiSolver:
    """$\\Phi$ in Z3: the three queries the consistency harness asks.

    Two assertion sets, for the reason the module docstring gives: `full` is
    every constraint of $\\Phi$, `facts` is $\\Phi$ without the checks. Both
    carry the product facts of §3.4; a query that answers `unknown` with them
    is retried without them, because they are the only nonlinear assertions
    and dropping them can only loosen the answer.
    """

    def __init__(self, record: dict[str, Any], budget_seconds: float) -> None:
        self.dag = TermDag.from_json(record["terms"])
        self.arena = Z3Arena(self.dag)
        constraints = record["constraints"]
        self.full = [self.arena[c["formula"]] for c in constraints]
        self.facts = [
            self.arena[c["formula"]] for c in constraints if c["kind"] != "check"
        ]
        self.products = [self._product(p) for p in record["products"]]
        self.budget_seconds = budget_seconds
        self.spent_seconds = 0.0
        #: One `z3.Solver` per assertion set, reused across queries. A
        #: circuit can have thousands of constraints and hundreds of
        #: queries; re-asserting them each time is what would dominate.
        self._solvers: dict[tuple[bool, bool], Any] = {}

    def _product(self, fact: dict[str, Any]) -> Any:
        left = self.arena.variables[int(fact["var"])]
        right: Any = z3.IntVal(1)
        for factor in fact["factors"]:
            right = right * self.arena[factor]
        return left == right

    @property
    def exhausted(self) -> bool:
        return self.spent_seconds >= self.budget_seconds

    def _charge(self, started: float) -> float:
        elapsed = time.time() - started
        self.spent_seconds += elapsed
        return elapsed

    def term(self, node_id: int) -> Any:
        """One node of the arena as a Z3 expression."""
        return self.arena[int(node_id)]

    def size_expression(self, dims: Sequence[int]) -> Any:
        """$\\prod_k\\mathrm{dims}[k]$ in Z3.

        The product is nonlinear as soon as two axes vary, which is why §3.4
        gives it a variable $\\sigma$ and a product fact rather than a term.
        Asking Z3 for the product directly lets those facts do their work.
        """
        expression: Any = z3.IntVal(1)
        for node_id in dims:
            expression = expression * self.arena[int(node_id)]
        return expression

    def conjunction(self, node_ids: Iterable[int]) -> Any:
        """The Boolean nodes `node_ids` conjoined, `true` when there are none."""
        terms = [self.arena[int(n)] for n in node_ids]
        return z3.And(*terms) if terms else z3.BoolVal(True)

    def bound(self, expression: Any, psi: Any, limit: int) -> "Answer":
        """Is `expression` at most `limit` under $\\Phi\\wedge\\psi$, and how far?

        Two queries, in that order, because they answer two different
        questions. The *pass condition* of §3.2 is only "phase 0's bound does
        not exceed the baseline's", which is one satisfiability query —
        `Φ ∧ ψ ∧ expr > limit` unsatisfiable — and stays inside the linear
        fragment whenever the expression does. The *tightening list* wants
        the exact maximum, which needs `z3.Optimize` and, on a nonlinear
        objective, is the query that gives up first. So a failure to optimise
        costs a `within` verdict, not a spurious violation.
        """
        first = self.exceeds(expression, psi, limit)
        if first.verdict != "within":
            return first
        best = self.maximize_expr(expression, psi)
        seconds = first.seconds + best.seconds
        if best.verdict != "bounded":
            return Answer("within", None, seconds, best.note)
        decided = "equal" if best.value == limit else "tighter"
        return Answer(decided, best.value, seconds, first.note)

    def exceeds(self, expression: Any, path: Any, limit: int) -> "Answer":
        """`within`, `looser` or `unknown` for `expr <= limit` under $\\Phi$."""
        started = time.time()
        goal = expression > z3.IntVal(limit)
        for solver, note in self._attempts(checks=True):
            answer = solver.check(path, goal)
            if answer == z3.unsat:
                return Answer("within", None, self._charge(started), note)
            if answer == z3.sat:
                return Answer("looser", None, self._charge(started), note)
        return Answer("unknown", None, self._charge(started), "Z3 timed out")

    def maximize_expr(self, expression: Any, psi: Any) -> "Answer":
        """The maximum of `expression` under $\\Phi\\wedge\\psi$; the verdict
        is `bounded`, `unbounded`, `unsatisfiable` or `unknown`."""
        started = time.time()
        for assertions, note in self._assertion_sets(checks=True):
            optimizer = z3.Optimize()
            optimizer.set("timeout", QUERY_TIMEOUT_MS)
            optimizer.add(*assertions, psi)
            handle = optimizer.maximize(expression)
            answer = optimizer.check()
            if answer == z3.unsat:
                return Answer("unsatisfiable", None, self._charge(started), note)
            if answer == z3.sat:
                upper = optimizer.upper(handle)
                if z3.is_int_value(upper):
                    return Answer("bounded", upper.as_long(), self._charge(started), note)
                return Answer("unbounded", None, self._charge(started), str(upper))
        return Answer("unknown", None, self._charge(started), "Z3 timed out")

    def entails(self, formula_id: int, path_id: int, facts_only: bool) -> "Answer":
        """Is `formula` true under the assertions and the path condition?

        `entailed` (the negation is unsatisfiable), `refutable` (the negation
        has a model), `vacuous` (the path condition itself is unsatisfiable)
        or `unknown`.
        """
        started = time.time()
        goal = z3.Not(self.arena[formula_id])
        path = self.arena[path_id]
        for solver, note in self._attempts(checks=not facts_only):
            if solver.check(path) == z3.unsat:
                return Answer("vacuous", None, self._charge(started), note)
            answer = solver.check(path, goal)
            if answer == z3.unsat:
                return Answer("entailed", None, self._charge(started), note)
            if answer == z3.sat:
                return Answer("refutable", None, self._charge(started), note)
        return Answer("unknown", None, self._charge(started), "Z3 timed out")

    def is_constant(self, term_id: int, path_id: int) -> "Answer":
        """Does $\\Phi\\wedge\\mathrm{path}$ pin the term to one value?"""
        started = time.time()
        term = self.arena[term_id]
        path = self.arena[path_id]
        solver = self._solver(checks=True, products=bool(self.products))
        if solver.check(path) != z3.sat:
            return Answer("vacuous", None, self._charge(started))
        witness = solver.model().eval(term, model_completion=True)
        answer = solver.check(path, term != witness)
        if answer == z3.unsat:
            return Answer(f"constant {witness}", None, self._charge(started))
        if answer == z3.sat:
            return Answer("not_constant", None, self._charge(started))
        return Answer("unknown", None, self._charge(started), "Z3 timed out")

    def _assertion_sets(self, checks: bool) -> Iterable[tuple[list[Any], str | None]]:
        """The assertion sets to try, in order: with the product facts of
        §3.4, then without. :meth:`_attempts` is the same list as cached
        solvers, and both are derived from :meth:`_note_of` so that the two
        cannot disagree about what to try.
        """
        base = self.full if checks else self.facts
        for products in self._product_attempts():
            yield ([*base, *self.products] if products else list(base)), self._note_of(products)

    def _attempts(self, checks: bool) -> Iterable[tuple[Any, str | None]]:
        """The same sets as :meth:`_assertion_sets`, as cached solvers."""
        for products in self._product_attempts():
            yield self._solver(checks, products), self._note_of(products)

    def _product_attempts(self) -> list[bool]:
        """Whether to assert the product facts, per attempt: with them first
        when there are any, then without.
        """
        return [True, False] if self.products else [False]

    def _note_of(self, products: bool) -> str | None:
        """Why an answer from this attempt is weaker than it could be."""
        return None if products or not self.products else (
            "answered without the product facts of §3.4"
        )

    def _solver(self, checks: bool, products: bool) -> Any:
        """One `z3.Solver` per assertion set, built once and reused."""
        cached = self._solvers.get((checks, products))
        if cached is None:
            cached = z3.Solver()
            cached.set("timeout", QUERY_TIMEOUT_MS)
            cached.add(*(self.full if checks else self.facts))
            if products:
                cached.add(*self.products)
            self._solvers[(checks, products)] = cached
        return cached
