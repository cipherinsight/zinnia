"""The adequacy check of `revision/phase0/acceptance.md` §2.5.

Given phase 0's record for a circuit and one reference run of the same circuit
on the same inputs (`refeval.py`), this module builds $\\theta(u)$ and applies
`definition.md` §7:

1. the run succeeded: every array descriptor's `dims` evaluates to the actual
   shape, every `defcond` and every $\\psi$ is true, and every constraint of
   $\\Phi$ holds under $\\theta(u)$;
2. the run failed: some check $\\varphi_k \\Rightarrow \\mathrm{defcond}_k$
   evaluates to false.

# Building $\\theta(u)$ (§2.3)

Every variable of $\\Phi$ carries its provenance, and each provenance says
which recorded value gives the variable its actual value:

| Provenance | $\\theta(u)$ |
|---|---|
| `input_length` | the actual length of that input axis |
| `shape_symbol` | the actual value of the scalar input or axis of that name |
| `count` | the elements the mask at the key selects; for `E-Comp`'s count, keyed by the comprehension itself, the length of what it kept |
| `unique_count`, `value_sized` | the leading extent of the value at the key |
| `data_scalar` | the number recorded at the key |
| `data_bool` | the truth of the value at the key |
| `definedness` | whether the run produced a value at the key at all: the witness stands for "the operation was defined" (`E-Take`'s `inrange`, an `E-Map` domain condition, `list.remove`) |
| `size_product` | the product fact of §3.4, evaluated |

A variable whose key the run never reached — an untaken branch, an iteration
of a bounded loop the data did not need — is given `0` / `false`. That is
inside the range §3.4 records for every kind of variable the corpus produces,
and every check that mentions it is guarded by a $\\varphi$ that is false, so
the choice cannot turn a sound record into a reported mismatch. Should it ever
do so anyway, the safety net is `_classify`: a mismatch whose term mentions a
defaulted variable is classified `unresolved_variable` and reported apart from
the defects.

# Cost

A circuit's term arena is evaluated once per draw, in one upward sweep over
the closure of everything the check reads (`TermDag.closure`). The closure is
computed once per circuit, in `Phase0Record.of`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any, Iterable

from testing.phase0.refeval import Observed, ParsedKey, Trace
from testing.phase0.terms import TermDag, UnboundVariable

#: How a mismatch is classified in the report. `rule_defect` is the only class
#: that counts against phase 0; the others say why the harness cannot decide.
RULE_DEFECT = "rule_defect"
#: The verdict rests on a variable whose value the run never produced.
UNRESOLVED_VARIABLE = "unresolved_variable"
#: Δ has a key the reference run never produced, outside any unrolled loop.
KEY_MISMATCH = "key_mismatch"
#: Δ has a key from an unrolled iteration the run's data did not need. Phase 0
#: unrolls to a cap and a real run stops earlier (§6.2), so this is expected
#: whenever the iteration's guard is not part of the value's $\\psi$.
UNROLLED_BEYOND_RUN = "unrolled_beyond_run"
#: The run failed at a limit of NumPy's own representation rather than at a
#: condition on the program's shapes; see :data:`NUMPY_LIMIT_ERRORS`.
NUMPY_LIMIT = "numpy_limit"
#: The run failed with a `TypeError`: the program is ill-typed, which the
#: front end diagnoses (`coverage.md` §4) and which §1 does not make phase 0's
#: business — it models shapes. Every shape condition raises `ValueError` or
#: `IndexError`, so the two cannot be confused.
ILL_TYPED = "ill_typed"

#: NumPy errors that report a limit of the reference implementation, not a
#: shape condition of the source: an extent or a size that does not fit
#: NumPy's `intp`. `np.reshape(np.zeros(0), (-1, 2**62))` raises the first of
#: them, although the shape it names — `(0, 2**62)` — has size 0 and is what
#: `definition.md` §5.6 gives the operation. Phase 0 models shapes, not the
#: machine limits of the evaluator the harness compares against, so a failure
#: of this kind is reported apart from the rule defects.
NUMPY_LIMIT_ERRORS = (
    "array is too big",
    "Maximum allowed dimension exceeded",
    "Maximum allowed shape exceeded",
    "exceeds maximum array size",
)

#: `cannot reshape array of size S into shape (d_0,d_1,…)`, with the target
#: NumPy prints back. The same message covers a genuine size mismatch and an
#: `intp` overflow in the target's product, so the extents have to decide
#: which of the two it is (:func:`_target_overflows_intp`).
_RESHAPE_ERROR = re.compile(r"cannot reshape array of size \d+ into shape \(([^)]*)\)")

#: The largest value NumPy's `intp` holds on a 64-bit platform.
_INTP_MAX = 2**63 - 1


@dataclass(frozen=True)
class Mismatch:
    """One disagreement between phase 0's record and the reference run."""

    key: str
    name: str | None
    kind: str
    expected: str
    actual: str
    classification: str = RULE_DEFECT
    #: The term the verdict rests on, when there is one. `_classify` reads its
    #: variables to tell a defect from an undecidable case.
    root: int | None = None

    def __str__(self) -> str:
        subject = f"{self.name or '<anonymous>'} at {self.key}"
        return (f"[{self.classification}] {self.kind}: {subject}: "
                f"expected {self.expected}, got {self.actual}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "kind": self.kind,
            "expected": self.expected,
            "actual": self.actual,
            "classification": self.classification,
        }


@dataclass
class DrawCheck:
    """The verdict on one draw."""

    #: `True` when the reference run succeeded.
    succeeded: bool
    mismatches: list[Mismatch] = field(default_factory=list)
    #: Set when the draw violates a `scalar_requires` fact, i.e. is not an
    #: admitted input after all (`acceptance.md` §2.4).
    inadmissible: bool = False
    #: Variables whose key the run never reached; see the module docstring.
    defaulted: set[int] = field(default_factory=set)

    @property
    def passed(self) -> bool:
        return not self.mismatches and not self.inadmissible


@dataclass
class Phase0Record:
    """One circuit's phase 0 output, prepared for repeated checking."""

    analysis: dict[str, Any]
    dag: TermDag
    #: Every node the check reads, sorted and closed under children.
    order: list[int]
    #: Product facts of §3.4, keyed by the variable they define.
    products: dict[int, dict[str, Any]]
    #: The `size_product` variables, which are resolved from those facts.
    size_products: list[dict[str, Any]]

    @staticmethod
    def of(analysis: dict[str, Any]) -> "Phase0Record":
        dag = TermDag.from_json(analysis["terms"])
        roots = list(_roots(analysis))
        return Phase0Record(
            analysis=analysis,
            dag=dag,
            order=dag.closure(roots),
            products={int(f["var"]): f for f in analysis.get("products", [])},
            size_products=[v for v in analysis["variables"]
                           if v["origin"]["kind"] == "size_product"],
        )


def _roots(analysis: dict[str, Any]) -> Iterable[int]:
    """The nodes the check evaluates: every constraint and every field of
    every descriptor, product factors included.
    """
    for constraint in analysis["constraints"]:
        yield int(constraint["formula"])
    for fact in analysis.get("products", []):
        yield from (int(factor) for factor in fact["factors"])
    for entry in analysis["descriptors"]:
        yield from _descriptor_roots(entry["desc"])


def _descriptor_roots(desc: dict[str, Any]) -> Iterable[int]:
    if desc["kind"] == "composite":
        for element in desc.get("elems") or []:
            yield from _descriptor_roots(element)
        return
    for field_name in ("defcond", "psi"):
        if desc.get(field_name) is not None:
            yield int(desc[field_name])
    for term in desc.get("dims") or []:
        yield int(term)
    if desc.get("term") is not None:
        yield int(desc["term"]["id"])


def check(record: Phase0Record, trace: Trace) -> DrawCheck:
    """Apply `acceptance.md` §2.5 to one reference run."""
    theta, defaulted, ambiguous = theta_of(record, trace)
    theta = _disambiguate(record, theta, ambiguous)
    values = record.dag.valuate(record.order, theta)
    verdict = DrawCheck(succeeded=trace.ok, defaulted=defaulted)
    if trace.ok:
        verdict.mismatches.extend(_check_descriptors(record, values, trace))
        verdict.mismatches.extend(_check_constraints(record, values, verdict))
    else:
        verdict.mismatches.extend(_check_failure(record, values))
    _classify(verdict, record, trace)
    return verdict


# ---------------------------------------------------------------------------
# theta(u)
# ---------------------------------------------------------------------------


def theta_of(
    record: Phase0Record, trace: Trace
) -> tuple[dict[int, Any], set[int], list[tuple[int, list[int]]]]:
    """$\\theta(u)$, the variables that had to be defaulted, and the ones whose
    value the recorded shape does not pin down.

    The ambiguous case is `E-Reshape`'s declared axis (`implementation.md` §8):
    $\\ell$ stands for *one* axis of the result and the export does not say
    which, so every distinct extent of the observed shape is a candidate and
    `_disambiguate` picks the one $\\Phi$ agrees with.
    """
    theta: dict[int, Any] = {}
    defaulted: set[int] = set()
    ambiguous: list[tuple[int, list[int]]] = []
    for variable in record.analysis["variables"]:
        var_id = int(variable["id"])
        if variable["origin"]["kind"] == "size_product":
            continue
        value = _value_of(variable, trace)
        if value is None:
            defaulted.add(var_id)
            theta[var_id] = False if variable["sort"] == "Bool" else 0
            continue
        theta[var_id] = value
        candidates = _candidates(variable, trace)
        if candidates is not None:
            ambiguous.append((var_id, candidates))
    _resolve_products(record, theta, defaulted)
    return theta, defaulted, ambiguous


def _candidates(variable: dict[str, Any], trace: Trace) -> list[int] | None:
    """The extents a `value_sized` variable could stand for, when the value it
    names has more than one axis. `None` when there is nothing to choose.
    """
    if variable["origin"]["kind"] != "value_sized":
        return None
    observed = trace.values.get(variable["origin"]["value"]["key"], None)
    if observed is None or observed.kind != "array" or not observed.shape:
        return None
    extents = sorted(set(int(n) for n in observed.shape))
    return extents if len(extents) > 1 else None


def _disambiguate(
    record: Phase0Record,
    theta: dict[int, Any],
    ambiguous: list[tuple[int, list[int]]],
) -> dict[int, Any]:
    """Pick, for each ambiguous variable, the candidate that satisfies the
    most constraints of $\\Phi$. Greedy and one variable at a time: two
    declared reshape axes in one program never constrain each other.
    """
    for var_id, candidates in ambiguous:
        best, best_score = theta[var_id], -1
        for candidate in candidates:
            theta[var_id] = candidate
            score = _satisfied(record, theta)
            if score > best_score:
                best, best_score = candidate, score
        theta[var_id] = best
    if ambiguous:
        _resolve_products(record, theta, set())
    return theta


def _satisfied(record: Phase0Record, theta: dict[int, Any]) -> int:
    """How many constraints of $\\Phi$ hold under `theta`.

    The product facts depend on the candidate, so they are recomputed first.
    """
    _resolve_products(record, theta, set())
    values = record.dag.valuate(record.order, theta)
    return sum(1 for c in record.analysis["constraints"]
               if values[int(c["formula"])])


def _value_of(variable: dict[str, Any], trace: Trace) -> Any:
    """The actual value of one variable, or `None` when the run did not
    reach the value it names (see the module docstring's table).
    """
    origin = variable["origin"]
    kind = origin["kind"]
    if kind == "input_length":
        return _input_axis(trace, origin["input"], int(origin["axis"]))
    if kind == "shape_symbol":
        return _shape_symbol(trace, origin["symbol"])
    located = origin.get("value") or origin.get("mask") or {}
    observed = trace.values.get(located.get("key", ""))
    if kind == "definedness":
        # A witness that the operation was defined: `E-Take`'s `inrange`
        # (§5.4), an `E-Map` domain condition (§5.3), `list.remove`'s "the
        # element is present" (§5.11). Φ does not model what they rest on, so
        # θ(u) reads them off the run: the value exists exactly when the
        # operation succeeded.
        return observed is not None
    if observed is None:
        return None
    if kind == "count":
        return observed.count
    if kind in ("unique_count", "value_sized"):
        return observed.leading_extent
    if kind == "data_bool":
        return _boolean(observed)
    if kind == "data_scalar":
        return _integer(observed)
    return None


def _integer(observed: Observed) -> int | None:
    if observed.kind != "scalar" or not isinstance(observed.number, (bool, int)):
        return None
    return int(observed.number)


def _boolean(observed: Observed) -> bool | None:
    """The truth of a recorded value, as $\\theta$ reads it.

    An array or composite at a data Boolean's key is a condition Python
    read for its truth — `if a:` on an array, or a mask used as one — and
    phase 0 cannot say which way it went, so the witness is true: it stands
    for "the condition was evaluated", the same reading a definedness
    witness gets.
    """
    if observed.kind in ("array", "composite"):
        return True
    if observed.kind == "scalar" and observed.number is not None:
        return bool(observed.number)
    return None


def _input_axis(trace: Trace, name: str, axis: int) -> int | None:
    """`n_x`: the actual extent of axis `axis` of the input named `name`."""
    observed = _input_named(trace, name)
    if observed is None or observed.shape is None or axis >= len(observed.shape):
        return None
    return int(observed.shape[axis])


def _input_named(trace: Trace, name: str) -> Observed | None:
    """Circuit inputs are keyed `|||k` in declaration order (§5a rule 1)."""
    if name not in trace.input_names:
        return None
    return trace.values.get(f"|||{trace.input_names.index(name)}")


def _shape_symbol(trace: Trace, symbol: str) -> int | None:
    """A `SymbolicNDArray` symbol: the scalar input of that name, or the axis
    the annotation labels with it (`definition.md` §5.1).
    """
    observed = _input_named(trace, symbol)
    if observed is not None and observed.kind == "scalar":
        return _integer(observed)
    return trace.shape_symbols.get(symbol)


def _resolve_products(
    record: Phase0Record, theta: dict[int, Any], defaulted: set[int]
) -> None:
    """$\\sigma = \\prod_k \\mathrm{dims}[k]$ (§3.4): the factors are terms,
    so the product is computed once every other variable has a value. A factor
    may itself be a $\\sigma$, so the list is swept until it stops changing.
    """
    for variable in record.size_products:
        theta.pop(int(variable["id"]), None)
    pending = list(record.size_products)
    for _ in range(len(pending) + 1):
        if not pending:
            break
        pending = [v for v in pending if not _resolve_one_product(record, theta, v)]
    for variable in pending:
        defaulted.add(int(variable["id"]))
        theta[int(variable["id"])] = 0


def _resolve_one_product(
    record: Phase0Record, theta: dict[int, Any], variable: dict[str, Any]
) -> bool:
    fact = record.products.get(int(variable["id"]))
    if fact is None:
        return False
    product = 1
    for factor in fact["factors"]:
        try:
            product *= int(record.dag.evaluate(int(factor), theta))
        except UnboundVariable:
            # A factor is itself a product not yet resolved; the sweep in
            # `_resolve_products` will come back to it. A plain `KeyError`
            # from a malformed record is a harness bug and must not be
            # read as "not resolvable yet".
            return False
    theta[int(variable["id"])] = product
    return True


# ---------------------------------------------------------------------------
# The three checks of §2.5
# ---------------------------------------------------------------------------


def _truth_of(values: dict[int, Any], node_id: Any) -> bool:
    """A Boolean field of a descriptor (`defcond`, `psi`)."""
    return True if node_id is None else bool(values[int(node_id)])


def _reached(values: dict[int, Any], desc: dict[str, Any]) -> bool:
    """Whether $\\Delta$ says the run computed this value.

    A composite has no $\\psi$ of its own (§3.3): the condition reaches its
    elements instead, so the composite was computed exactly when they were.
    An empty composite has nothing to read and counts as computed.
    """
    if desc["kind"] == "composite":
        return all(_reached(values, element) for element in desc["elems"])
    return _truth_of(values, desc.get("psi"))


def _check_descriptors(
    record: Phase0Record, values: dict[int, Any], trace: Trace
) -> list[Mismatch]:
    """G2: every descriptor of Δ against the value the run computed."""
    problems: list[Mismatch] = []
    for entry in record.analysis["descriptors"]:
        key = entry["key"]["key"]
        if entry["key"].get("step", 0):
            # An *intermediate* of `definition.md` §3.3 — the array an
            # operator stacks a composite operand into, a coordinate of
            # `unravel_index`, the element index of an unrolled iteration,
            # the comparison a division records for `defcond_div`. The
            # reference evaluator mints no key for one, because the source
            # never wrote it, so there is no observed value to compare
            # against; its shape reaches this harness through the operands
            # of the rules that read it (`acceptance.md` §3).
            continue
        observed = trace.values.get(key)
        problems.extend(
            _check_entry(values, entry["desc"], observed, key, entry.get("name"))
        )
    return problems


def _check_entry(
    values: dict[int, Any],
    desc: dict[str, Any],
    observed: Observed | None,
    key: str,
    name: str | None,
) -> list[Mismatch]:
    """A value the run did not compute must have a false $\\psi$; one it did
    compute must have a true $\\psi$ and a matching descriptor.

    An entry with no descriptor content — a dtype or module name, `None` —
    is not a value (`implementation.md` §5a) and has nothing to compare.
    """
    if desc["kind"] in ("class", "none"):
        return []
    reached = _reached(values, desc)
    if observed is None:
        if reached:
            return [Mismatch(key, name, "unreached value", "psi false",
                             "the run never computed this value", KEY_MISMATCH)]
        return []
    if not reached:
        return [Mismatch(key, name, "psi", "true",
                         "false, though the run computed the value",
                         root=desc.get("psi"))]
    return _compare(values, desc, observed, key, name)


def _compare(
    values: dict[int, Any],
    desc: dict[str, Any],
    observed: Observed,
    key: str,
    name: str | None,
) -> list[Mismatch]:
    """One descriptor against one observed value."""
    problems: list[Mismatch] = []
    if not _truth_of(values, desc.get("defcond")):
        problems.append(Mismatch(key, name, "defcond", "true", "false",
                                 root=desc.get("defcond")))
    kind = desc["kind"]
    if kind == "array":
        problems.extend(_compare_array(values, desc, observed, key, name))
    elif kind == "scalar":
        problems.extend(_compare_scalar(values, desc, observed, key, name))
    elif kind == "composite":
        problems.extend(_compare_composite(values, desc, observed, key, name))
    return problems


def _compare_array(
    values: dict[int, Any],
    desc: dict[str, Any],
    observed: Observed,
    key: str,
    name: str | None,
) -> list[Mismatch]:
    """`dims` must evaluate to the actual logical shape (§7.2).

    A rank-1 array and a flat composite of the same length denote the same
    value in phase 0 — `as_array` stacks a composite and `tolist` unstacks one
    — so a composite is read as its length here.
    """
    dims = tuple(int(values[int(t)]) for t in desc["dims"])
    shape = observed.shape
    if observed.kind == "composite" and len(dims) == 1 and observed.elems is not None:
        shape = (len(observed.elems),)
    if observed.kind == "scalar" and not dims:
        shape = ()
    if shape is None or tuple(shape) != dims:
        pretty = ", ".join(desc.get("dims_pretty") or [])
        return [Mismatch(key, name, "dims", f"[{pretty}] = {list(dims)}",
                         f"{observed.kind} {shape}",
                         root=int(desc["dims"][0]) if desc["dims"] else None)]
    return []


def _compare_scalar(
    values: dict[int, Any],
    desc: dict[str, Any],
    observed: Observed,
    key: str,
    name: str | None,
) -> list[Mismatch]:
    """A scalar of origin `shape` carries a term built from $S$ and other
    shape terms, so the term must evaluate to the value the run computed
    (§3.2). A data scalar's term is the fresh variable $\\theta$ just read off
    that value, so there is nothing to check.
    """
    term = desc.get("term")
    if desc.get("origin") != "shape" or term is None:
        return []
    if observed.kind != "scalar" or not isinstance(observed.number, (bool, int)):
        return []
    value = int(values[int(term["id"])])
    if value != int(observed.number):
        return [Mismatch(key, name, "shape term",
                         f"{desc.get('term_pretty')} = {value}",
                         str(int(observed.number)), root=int(term["id"]))]
    return []


def _compare_composite(
    values: dict[int, Any],
    desc: dict[str, Any],
    observed: Observed,
    key: str,
    name: str | None,
) -> list[Mismatch]:
    elems = desc.get("elems") or []
    if observed.kind == "array":
        # A composite of `k` elements and an array whose leading extent is `k`
        # denote the same value in phase 0, which :func:`_compare_array` reads
        # the other way round. `lst == other` is one such value: §5.11 gives
        # it a composite and NumPy gives the run an array. Only the leading
        # extent is comparable — an array's elements are numbers, not
        # descriptors — so that is what is checked.
        if observed.shape and observed.shape[0] == len(elems):
            return []
        return [Mismatch(key, name, "composite length", str(len(elems)),
                         f"array {observed.shape}")]
    if observed.kind != "composite" or observed.elems is None:
        return [Mismatch(key, name, "composite", f"{len(elems)} elements",
                         observed.kind)]
    if len(observed.elems) != len(elems):
        return [Mismatch(key, name, "composite length", str(len(elems)),
                         str(len(observed.elems)))]
    problems: list[Mismatch] = []
    for index, (element, seen) in enumerate(zip(elems, observed.elems)):
        problems.extend(_compare(values, element, seen, key,
                                 f"{name}[{index}]" if name else None))
    return problems


def _check_constraints(
    record: Phase0Record, values: dict[int, Any], verdict: DrawCheck
) -> list[Mismatch]:
    """G1: a successful run satisfies every constraint of $\\Phi$.

    A false `precondition` means the draw is outside the admitted domain
    (§2.4), which is a fact about the draw and not about phase 0.
    """
    problems: list[Mismatch] = []
    for constraint in record.analysis["constraints"]:
        if values[int(constraint["formula"])]:
            continue
        if constraint["kind"] == "precondition":
            verdict.inadmissible = True
            continue
        problems.append(
            Mismatch("<Phi>", constraint["provenance"], constraint["kind"],
                     "true", f"false: {constraint['pretty'][:160]}",
                     root=int(constraint["formula"]))
        )
    return problems


def _check_failure(record: Phase0Record, values: dict[int, Any]) -> list[Mismatch]:
    """G3: a failing run violates some check $\\varphi_k \\Rightarrow
    \\mathrm{defcond}_k$.
    """
    checks = [c for c in record.analysis["constraints"] if c["kind"] == "check"]
    for constraint in checks:
        if not values[int(constraint["formula"])]:
            return []
    return [
        Mismatch("<Phi>", None, "no violated check",
                 "some check false on a failing run",
                 f"all {len(checks)} checks hold")
    ]


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _classify(verdict: DrawCheck, record: Phase0Record, trace: Trace) -> None:
    """Separate the mismatches phase 0 is answerable for from the four
    harness-visible causes: a key from an iteration the run did not need, a
    variable the run could not resolve, a run that failed at a limit of
    NumPy's own representation, and a run that failed because the program is
    ill-typed.
    """
    reached = _reached_iterations(trace)
    verdict.mismatches = [
        replace(
            mismatch,
            classification=_classification_of(mismatch, verdict, record, trace, reached),
        )
        for mismatch in verdict.mismatches
    ]


def _classification_of(
    mismatch: Mismatch,
    verdict: DrawCheck,
    record: Phase0Record,
    trace: Trace,
    reached: dict[tuple[str, str], int],
) -> str:
    if mismatch.classification == KEY_MISMATCH:
        return (UNROLLED_BEYOND_RUN if _beyond_trip_count(mismatch.key, reached)
                else KEY_MISMATCH)
    if mismatch.classification != RULE_DEFECT:
        return mismatch.classification
    if mismatch.kind == "no violated check" and trace.ill_typed:
        return ILL_TYPED
    if _at_a_numpy_limit(mismatch, trace):
        return NUMPY_LIMIT
    if _mentions_defaulted(mismatch, verdict, record):
        return UNRESOLVED_VARIABLE
    return RULE_DEFECT


def _reached_iterations(trace: Trace) -> dict[tuple[str, str], int]:
    """The highest iteration index the run reached at each loop nesting.

    Keyed by `(call stack, enclosing iteration counters)`, which is as precise
    as the key string allows: two sibling loops at the same depth share an
    entry, and sharing can only make the harness more forgiving.
    """
    highest: dict[tuple[str, str], int] = {}
    for key in trace.values:
        nest = _loop_nest(key)
        if nest is None:
            continue
        bucket, index = nest
        highest[bucket] = max(highest.get(bucket, -1), index)
    return highest


def _beyond_trip_count(key: str, reached: dict[tuple[str, str], int]) -> bool:
    """Whether a missing key names an iteration the run never performed."""
    nest = _loop_nest(key)
    if nest is None:
        return False
    bucket, index = nest
    return index > reached.get(bucket, -1)


def _loop_nest(key: str) -> tuple[tuple[str, str], int] | None:
    """The innermost loop of `key` as `(bucket, iteration index)`, or `None`
    when the key is not inside a loop at all.
    """
    parsed = ParsedKey.parse(key)
    if not parsed.iters:
        return None
    outer = ".".join(str(n) for n in parsed.iters[:-1])
    return ((".".join(parsed.calls), outer), parsed.iters[-1])


def _at_a_numpy_limit(mismatch: Mismatch, trace: Trace) -> bool:
    """Whether the run failed at one of NumPy's representation limits.

    Narrow on purpose: only the "no check was violated" verdict of a *failing*
    run can be excused this way — a wrong shape on a successful run never is —
    and only for the errors :data:`NUMPY_LIMIT_ERRORS` names plus a `reshape`
    whose target NumPy cannot represent.
    """
    if mismatch.kind != "no violated check" or trace.error is None:
        return False
    if any(text in trace.error for text in NUMPY_LIMIT_ERRORS):
        return True
    return _target_overflows_intp(trace.error)


def _target_overflows_intp(error: str) -> bool:
    """Whether a `cannot reshape` message names a target NumPy cannot
    represent, rather than one whose size disagrees with the input's.

    NumPy multiplies the target's known extents in `intp` and refuses when
    that overflows, with the same message it uses for a genuine size
    mismatch. §5.6 gives the operation a shape either way — `(8, 0, 2**60)`
    has size 0 — so the overflow is a limit of the evaluator and not of the
    record. A genuine mismatch cannot reach here: `defcond` is
    `size(a) = ∏ target`, which is false on such a run, and a violated check
    is what G3 asks for.
    """
    match = _RESHAPE_ERROR.search(error)
    if match is None:
        return False
    product = 1
    for entry in match.group(1).split(","):
        entry = entry.strip()
        # `newaxis` is the inferred `-1`, which contributes no factor.
        if entry in ("", "newaxis"):
            continue
        if not entry.lstrip("-").isdigit():
            return False
        product *= abs(int(entry))
    return product > _INTP_MAX


def _mentions_defaulted(
    mismatch: Mismatch, verdict: DrawCheck, record: Phase0Record
) -> bool:
    """Whether the term at fault mentions a variable the run never reached,
    which makes the verdict inconclusive rather than wrong.
    """
    if not verdict.defaulted or mismatch.root is None:
        return False
    return bool(record.dag.variables(int(mismatch.root)) & verdict.defaulted)
