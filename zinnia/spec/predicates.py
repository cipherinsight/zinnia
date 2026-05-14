"""Predicate marker symbols for use inside ``@requires`` lambdas.

Each name in this module is a *marker* — calling one at runtime raises
``RuntimeError``. They are recognized syntactically when the AST transformer
walks a ``@requires(...)`` decorator's lambda body.

Importing the names lets the user's IDE autocomplete them and surfaces typos
(``from zinnia.spec.predicates import nzz`` fails at import).

Categories:
  - cardinality:     ``nnz``, ``popcount``
  - ordering:        ``is_sorted``, ``is_monotone_nondecreasing``, ``max_run``
  - permutation:     ``is_permutation``, ``cycle_count``, ``fixed_point_count``
  - element-wise:    ``forall``, ``lt_bound``, ``eq_const``, ``in_range``,
                     ``is_nonneg``
"""


class _PredicateMarker:
    """A non-callable marker that records its name for the AST transformer."""

    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            f"predicate `{self.name}` is only valid inside a `@requires(...)` "
            f"lambda; do not call it at runtime."
        )

    def __repr__(self) -> str:
        return f"<zinnia.spec.predicate {self.name}>"


# Registered predicate names. The AST transformer validates against this set
# at decoration time; unknown predicate names raise a frontend error.
# User-surface predicate names — each has a callable marker symbol
# exported from this module. The AST extractor recognises these as
# valid identifiers inside `@requires(lambda ...)`.
_USER_PREDICATE_NAMES = frozenset({
    # cardinality
    "nnz",
    "popcount",
    # ordering
    "is_sorted",
    "is_monotone_nondecreasing",
    "max_run",
    # permutation
    "is_permutation",
    "cycle_count",
    "fixed_point_count",
    # element-wise outer + closed inner predicates
    "forall",
    "lt_bound",
    "eq_const",
    "in_range",
    "is_nonneg",
})

# Synthesized kinds emitted by the AST extractor when flattening
# `forall(arr, P)` to a single IR atom. These appear in IR dumps and
# in the Rust registry; users do **not** write them directly. See
# `_FORALL_INNER_PREDICATES` in
# `zinnia/compile/transformer/_precondition.py`.
_SYNTHESIZED_KINDS = frozenset({
    "forall_is_nonneg",
    "forall_lt_bound",
    "forall_eq_const",
    "forall_in_range",
})

# Combined set of all valid IR kind names. The compile-time validator
# in `_precondition._build_spec` checks against this.
PREDICATE_NAMES = _USER_PREDICATE_NAMES | _SYNTHESIZED_KINDS


# Marker instances, one per name.
nnz = _PredicateMarker("nnz")
popcount = _PredicateMarker("popcount")

is_sorted = _PredicateMarker("is_sorted")
is_monotone_nondecreasing = _PredicateMarker("is_monotone_nondecreasing")
max_run = _PredicateMarker("max_run")

is_permutation = _PredicateMarker("is_permutation")
cycle_count = _PredicateMarker("cycle_count")
fixed_point_count = _PredicateMarker("fixed_point_count")

forall = _PredicateMarker("forall")
lt_bound = _PredicateMarker("lt_bound")
eq_const = _PredicateMarker("eq_const")
in_range = _PredicateMarker("in_range")
is_nonneg = _PredicateMarker("is_nonneg")


__all__ = [
    "PREDICATE_NAMES",
    "nnz", "popcount",
    "is_sorted", "is_monotone_nondecreasing", "max_run",
    "is_permutation", "cycle_count", "fixed_point_count",
    "forall", "lt_bound", "eq_const", "in_range", "is_nonneg",
]
