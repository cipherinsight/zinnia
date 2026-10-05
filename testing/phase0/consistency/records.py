"""The rows of the consistency report, and how the two records are read.

`revision/phase0/acceptance.md` §3.1 fixes what each comparison answers;
these are the dataclasses that hold the answers, plus the three readers every
comparison starts from: the descriptor leaves of phase 0's record
(:func:`descriptor_leaves`), the arrays the program allocates
(:func:`allocated_arrays`) and the input descriptors by name
(:func:`input_descriptors`).

Nothing here calls Z3 or reads the baseline.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any

from testing.phase0.phi_solver import PhiSolver
from testing.phase0.terms import TermDag

#: Distinct checks answered per circuit. An unrolled loop nest records one
#: check per iteration per operation, so a program can have tens of
#: thousands; answering them all would take hours and say nothing the first
#: few hundred do not. The rest are reported as `not_attempted`.
CHECK_BUDGET = int(os.environ.get("ZINNIA_PHASE0_CHECK_BUDGET", "200"))

#: How much of a term's pretty form a report row keeps.
PRETTY_CUTOFF = 240

#: Unmatched arrays a row lists one by one before it groups them.
UNMATCHED_ROWS = 50


@dataclass(frozen=True)
class AxisBound:
    """Phase 0's bound on one axis against the baseline's.

    | Verdict | Meaning |
    |---|---|
    | `equal` | the maximum over $\\Phi\\wedge\\psi$ is the baseline's bound |
    | `tighter` | it is strictly smaller — a tightening, which §3.2 makes an output and not a condition |
    | `within` | $\\Phi$ does not admit a value above the baseline's bound, but Z3 could not pin the maximum |
    | `looser` | $\\Phi\\wedge\\psi$ admits a value above the baseline's bound, which fails |
    | `unknown` | Z3 answered neither |
    """

    axis: int
    baseline: int
    phase0: int | None
    expression: str
    verdict: str
    seconds: float = 0.0
    note: str | None = None

    @property
    def fails(self) -> bool:
        return self.verdict == "looser"


@dataclass(frozen=True)
class InputCheck:
    """One circuit parameter, phase 0's descriptor against the baseline's."""

    name: str
    baseline: str
    phase0: str
    verdict: str
    axes: list[AxisBound] = field(default_factory=list)
    detail: str | None = None

    #: Verdicts that fail `acceptance.md` §3.2.
    FAILING = ("shape_differs", "rank_differs", "kind_differs", "missing_in_phase0")

    @property
    def fails(self) -> bool:
        return self.verdict in self.FAILING or any(a.fails for a in self.axes)


@dataclass(frozen=True)
class StaticMatch:
    """One baseline memory segment against one phase 0 constant-dims array."""

    size: int
    baseline: str | None
    phase0: str | None
    #: `matched`, `dynamic_backing` (the segment holds a dynamic array's
    #: payload), `unmatched_baseline` or `unmatched_phase0`.
    verdict: str
    #: Arrays this row stands for; above one only in an aggregated row.
    count: int = 1


@dataclass(frozen=True)
class DynamicMatch:
    """One baseline dynamic envelope against one phase 0 non-constant array."""

    baseline: str | None
    phase0: str | None
    match: str  # param | order | unmatched_baseline | unmatched_phase0
    axes: list[AxisBound] = field(default_factory=list)
    detail: str | None = None
    #: Arrays this row stands for; above one only in an aggregated row.
    count: int = 1

    @property
    def fails(self) -> bool:
        return any(axis.fails for axis in self.axes)


@dataclass(frozen=True)
class ReturnCheck:
    """One row about the delivery set.

    Either one delivered array against the `allowed_meta` oracle (`member`,
    `not_member`, `envelope`, `not_applicable`, `unknown`) or, with
    `path == "payload"`, the whole delivery set's float envelope against the
    cells the baseline bound (`equal`, `tighter`, `within`, `looser`).

    Only `looser` fails. `not_member` is a disagreement with *NumPy* on the
    fixture the oracle runs, not with a shape the baseline answered, and
    `acceptance.md` §3.2's pass condition names three relations, none of them
    this one; §2.5 is where a disagreement with NumPy is a verdict. The rows
    are reported, and the report lists every one.
    """

    path: str
    dims: list[str]
    bounds: list[int | None]
    verdict: str
    detail: str | None = None

    @property
    def fails(self) -> bool:
        return self.verdict == "looser"


@dataclass(frozen=True)
class CheckResult:
    """Z3's answer to one check $\\Phi$ records: can its `defcond` be false?"""

    index: int
    provenance: str
    pretty: str
    answer: str
    seconds: float
    #: Checks this row stands for: identical `(path, defcond)` questions are
    #: answered once.
    occurrences: int = 1
    #: Distinct questions this row stands for; above one only in the
    #: aggregated `not_attempted` row.
    distinct: int = 1
    note: str | None = None

    @property
    def discharged(self) -> bool:
        """The check got a definite answer."""
        return self.answer not in ("unknown", "not_attempted")


@dataclass(frozen=True)
class FactSummary:
    """The baseline's own facts, and what of them is extractable."""

    baseline_enforced: dict[str, int]
    phase0_checks: int
    exact_shape_resolutions: list[str]
    unextractable: list[str]


@dataclass
class CircuitConsistency:
    """One row of `revision/phase0/results/consistency.json`."""

    program: str
    function: str
    status: str
    detail: str | None = None
    inputs: list[InputCheck] = field(default_factory=list)
    static_arrays: list[StaticMatch] = field(default_factory=list)
    dynamic_arrays: list[DynamicMatch] = field(default_factory=list)
    returns: list[ReturnCheck] = field(default_factory=list)
    checks: list[CheckResult] = field(default_factory=list)
    facts: FactSummary | None = None
    sizes: dict[str, int] = field(default_factory=dict)
    seconds: dict[str, float] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.program}::{self.function}"

    @property
    def tightenings(self) -> list[AxisBound]:
        """Every axis where phase 0's bound is strictly below the baseline's."""
        axes = [a for check in self.inputs for a in check.axes]
        axes += [a for match in self.dynamic_arrays for a in match.axes]
        return [axis for axis in axes if axis.verdict == "tighter"]

    @property
    def failures(self) -> list[str]:
        """The `acceptance.md` §3.2 violations, as messages naming the array."""
        out = []
        for check in self.inputs:
            if check.verdict in InputCheck.FAILING:
                out.append(f"input `{check.name}`: {check.verdict}: {check.detail}")
            out += [_axis_failure(f"input `{check.name}`", a) for a in check.axes if a.fails]
        for match in self.dynamic_arrays:
            out += [_axis_failure(str(match.phase0), a) for a in match.axes if a.fails]
        out += [
            f"return {check.path}: {check.verdict}, phase 0 says {check.bounds} "
            f"({check.detail})"
            for check in self.returns
            if check.fails
        ]
        return out

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _axis_failure(array: str, axis: AxisBound) -> str:
    """One §3.2 violation, naming the array, the axis and the term."""
    return (
        f"{array} axis {axis.axis}: Phi admits an extent above the baseline's "
        f"bound of {axis.baseline} ({axis.expression})"
    )


# ---------------------------------------------------------------------------
# Reading the two records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArrayEntry:
    """One leaf of $\\Delta$: an array, or a scalar as a rank-0 array.

    A scalar has an empty `dims` and therefore size 1, which is what makes
    the return-payload count below uniform over both. `kind` keeps the two
    apart everywhere the difference matters.
    """

    key: str
    path: str
    kind: str  # array | scalar
    dtype: str
    dims: list[int]
    dims_pretty: list[str]
    psi: int
    constant: list[int] | None

    @property
    def size(self) -> int | None:
        """$\\prod_k\\mathrm{dims}[k]$ when every extent is a literal."""
        if self.constant is None:
            return None
        size = 1
        for extent in self.constant:
            size *= extent
        return size


def _constant_dims(dag: TermDag, dims: list[int]) -> list[int] | None:
    """The literal extents, or `None` when any axis is a non-constant term."""
    values = []
    for node_id in dims:
        node = dag.nodes[int(node_id)]
        if node["op"] != "const":
            return None
        values.append(int(node["value"]))
    return values


def _walk_descriptor(
    dag: TermDag, desc: dict[str, Any], key: str, path: str, out: list[ArrayEntry]
) -> None:
    """Collect the array and scalar leaves inside one $\\Delta$ entry."""
    kind = desc["kind"]
    if kind == "composite":
        for index, element in enumerate(desc["elems"]):
            _walk_descriptor(dag, element, key, f"{path}[{index}]", out)
        return
    if kind not in ("array", "scalar"):
        return
    dims = [int(d) for d in desc["dims"]] if kind == "array" else []
    pretty = [cut(text) for text in desc["dims_pretty"]] if kind == "array" else []
    out.append(
        ArrayEntry(
            key=key,
            path=path,
            kind=kind,
            dtype=desc["dtype"],
            dims=dims,
            dims_pretty=pretty,
            psi=int(desc["psi"]),
            constant=_constant_dims(dag, dims),
        )
    )


def descriptor_leaves(
    record: dict[str, Any], dag: TermDag, delivery: bool
) -> list[ArrayEntry]:
    """Every array and scalar leaf of $\\Delta$, in traversal order.

    `delivery` selects the delivery set $R$ (the entries whose key is one of
    `record["delivery"]`) or its complement.
    """
    keys = {k["key"] for k in record["delivery"]}
    out: list[ArrayEntry] = []
    for entry in record["descriptors"]:
        key = entry["key"]["key"]
        if (key in keys) != delivery:
            continue
        name = entry.get("name") or key
        _walk_descriptor(dag, entry["desc"], key, name, out)
    return out


#: The producer rules that give an existing value a new key instead of making
#: a new value. An assignment binds its right-hand side, and the new version a
#: subscript write leaves behind is the array that was written into — §5.11
#: makes a write change no extent. An `alias` producer says the same thing for
#: a use of a name. :func:`allocated_arrays` folds all three into the value
#: they point at.
RELABELLING_RULES = frozenset({"S-Assign", "S-SetItem"})


def _relabelled(producer: dict[str, Any]) -> str | None:
    """The key this entry re-labels, or `None` when it is a value of its own."""
    if producer["kind"] == "alias":
        return producer["alias"]
    if producer["kind"] != "applied":
        return None
    rules = set(producer.get("rules") or [])
    operands = producer.get("operands") or []
    if rules and rules <= RELABELLING_RULES and operands:
        return operands[0]
    return None


def allocated_arrays(record: dict[str, Any], dag: TermDag) -> list[ArrayEntry]:
    """The array leaves of the values the program *makes*, once each, in
    traversal order.

    $\\Delta$ is complete — one entry per keyed value (`implementation.md`
    §5a) — so one array appears under several keys: the expression that built
    it, the name an assignment bound it to, every later use of that name, the
    version each write into it leaves behind. §3.1's array comparison asks
    which arrays the program allocates, so the keys that only re-label a
    value are folded into the value they point at
    (:data:`RELABELLING_RULES`), which keeps its place in the traversal and
    takes the source name, if any, that was bound to it. What is left is one
    entry per array the program makes: an input, an intermediate a rule built,
    a merge. The delivery set is excluded, as it is its own comparison.
    """
    delivery = {k["key"] for k in record["delivery"]}
    origin: dict[str, str] = {}
    name_of: dict[str, str] = {}
    maker: dict[str, dict[str, Any]] = {}
    for entry in record["descriptors"]:
        key = entry["key"]["key"]
        source = _relabelled(entry["producer"])
        root = origin.get(source, key) if source is not None else key
        origin[key] = root
        maker.setdefault(root, entry)
        if entry.get("name") and root not in name_of:
            name_of[root] = entry["name"]
    out: list[ArrayEntry] = []
    for root, entry in maker.items():
        if root in delivery:
            continue
        _walk_descriptor(dag, entry["desc"], root, name_of.get(root, root), out)
    return [entry for entry in out if entry.kind == "array"]


def input_descriptors(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The $\\Delta$ entry of each circuit parameter, by name.

    `implementation.md` §5a rule 1: input `k` is keyed with an empty statement
    path, no iterations, no calls and expression index `k`, and is recorded
    before the walk, so the first entry with such a key is the parameter.
    """
    delivery = {k["key"] for k in record["delivery"]}
    out: dict[str, dict[str, Any]] = {}
    for entry in record["descriptors"]:
        key = entry["key"]
        if key["stmt_path"] or key["iters"] or key["calls"]:
            continue
        name = entry.get("name")
        if name is None or name in out or key["key"] in delivery:
            continue
        out[name] = entry["desc"]
    return out


def cut(text: str) -> str:
    """A pretty form short enough for a report row."""
    return text if len(text) <= PRETTY_CUTOFF else text[: PRETTY_CUTOFF - 1] + "…"


def axis_by_solver(
    dag: TermDag,
    solver: PhiSolver | None,
    node_id: int,
    psi: int,
    limit: int,
    axis: int,
) -> AxisBound:
    """The axis verdict from $\\max(\\mathrm{dims}[k])$ over $\\Phi\\wedge\\psi$.

    Rows (a) and (c) both fill an [`AxisBound`] this way — an input's bounded
    axis against the annotation, a dynamic array's axis against the
    baseline's envelope — so the query lives beside the row it fills.
    """
    expression = cut(dag.pretty(int(node_id)))
    if solver is None or solver.exhausted:
        note = "no solver" if solver is None else "the circuit's Z3 budget ran out"
        return AxisBound(axis, limit, None, expression, "unknown", note=note)
    answer = solver.bound(solver.term(node_id), solver.term(psi), limit)
    return AxisBound(
        axis, limit, answer.value, expression, answer.verdict, answer.seconds, answer.note
    )
