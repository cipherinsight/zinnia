"""The micro-benchmark kernels of the array-fact constructions, one per rule.

Each rule is a kernel template over a size `n`, its inputs (a function of `n`
and a seeded generator), and the configurations it is measured under: `with`
(the compiler), `without` (the same compiler with the rule off, through the
measurement switches of `docs/ablations.md`), and for rules with alternative
constructions one configuration per construction forced. The kernels state the
fact the rule keys on the way a program would (`@requires`), and compute what
NumPy computes; the runner checks every compiled circuit's mock outputs
against NumPy.

The rules are general: none of these kernels is in the evaluation corpus (the
corpus barely exercises array facts, `eval-redo/layer/array-fact-survey.md`),
so these sweeps are how the effect of each rule is shown.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

#: The switch that turns every array claim off (`ZINNIA_ABLATE_ARRAY_FACTS`).
WITHOUT_FACTS = {"ZINNIA_ABLATE_ARRAY_FACTS": "1"}


@dataclass
class Rule:
    name: str
    description: str
    signature: str  # format string over n
    body: str  # format string over n
    requires: list[str]
    inputs: Callable[[int, np.random.Generator], list]
    configs: dict[str, dict[str, str]] = field(default_factory=lambda: {"with": {}, "without": WITHOUT_FACTS})
    sizes: tuple[int, ...] = (8, 16, 32, 64, 128)

    def source(self, n: int) -> str:
        head = "from zinnia import *\nimport numpy as np\n\n@zk_circuit\n"
        head += "".join(f"@requires({r})\n" for r in self.requires)
        body = "".join(f"    {line}\n" for line in self.body.format(n=n).split("\n"))
        return head + f"def kernel({self.signature.format(n=n)}):\n" + body


def ints(lo: int, hi: int):
    return lambda n, r: r.integers(lo, hi + 1, n).astype(np.int64)


def sorted_ints(lo: int, hi: int):
    """Non-decreasing integers with ties."""
    return lambda n, r: np.sort(r.integers(lo, hi + 1, n)).astype(np.int64)


ARR = 'a: Integer[Array, "{n}"]'
RANGE = "lambda a: forall(a, in_range(0, 255))"
SORTED = "lambda a: is_sorted(a)"

RULES: dict[str, Rule] = {}


def rule(**kw) -> None:
    r = Rule(**kw)
    RULES[r.name] = r


# ── Element ranges ──────────────────────────────────────────────────────
rule(name="range_compare", description="a < b of two arrays checked into [0, 255]: range realization, narrow comparison",
     signature=ARR + ', b: Integer[Array, "{n}"]', body="return a < b",
     requires=["lambda a, b: forall(a, in_range(0, 255))", "lambda a, b: forall(b, in_range(0, 255))"],
     inputs=lambda n, r: [ints(0, 255)(n, r), ints(0, 255)(n, r)])
rule(name="range_compare_float", description="a < 0.5 of floats checked into [0.0, 1.0]: range of the codes",
     signature='a: Float[Array, "{n}"]', body="return a < 0.5", requires=["lambda a: forall(a, in_range(0.0, 1.0))"],
     inputs=lambda n, r: [r.uniform(0, 1, n)])
rule(name="range_sum_mean", description="sum and mean of an array in [0, 255]: bounded reductions",
     signature=ARR, body="return a.sum(), a.mean()", requires=[RANGE], inputs=lambda n, r: [ints(0, 255)(n, r)])
rule(name="range_max", description="np.max of an array in [0, 255]: the hinted maximum bounded into the range",
     signature=ARR, body="return np.max(a)", requires=[RANGE], inputs=lambda n, r: [ints(0, 255)(n, r)])
rule(name="range_sort", description="np.sort of an array in [0, 255]: narrow comparators and pins",
     signature=ARR, body="return np.sort(a)", requires=[RANGE], inputs=lambda n, r: [ints(0, 255)(n, r)])
rule(name="range_argmax", description="np.argmax of an array in [0, 255]: narrow running selects",
     signature=ARR, body="return np.argmax(a)", requires=[RANGE], inputs=lambda n, r: [ints(0, 255)(n, r)])
rule(name="clip_identity", description="np.clip(a, 0, 255) of an array in [0, 255]: the identity",
     signature=ARR, body="return np.clip(a, 0, 255)", requires=[RANGE], inputs=lambda n, r: [ints(0, 255)(n, r)])
rule(name="where_decided", description="np.where(a >= 0, a, -a) of an array in [0, 255]: the comparison decided",
     signature=ARR, body="return np.where(a >= 0, a, -a)", requires=[RANGE], inputs=lambda n, r: [ints(0, 255)(n, r)])

# ── Sorted inputs ───────────────────────────────────────────────────────
rule(name="sorted_max_min", description="np.max and np.min of a sorted array: boundary reads",
     signature=ARR, body="return np.max(a), np.min(a)", requires=[SORTED], inputs=lambda n, r: [sorted_ints(0, 9)(n, r)])
rule(name="sorted_median", description="np.median of a sorted array: middle reads",
     signature=ARR, body="return np.median(a)", requires=[SORTED], inputs=lambda n, r: [sorted_ints(0, 9)(n, r)])
rule(name="sorted_argmax", description="np.argmax and np.argmin of a sorted array (ties: the first maximum)",
     signature=ARR, body="return np.argmax(a), np.argmin(a)", requires=[SORTED],
     inputs=lambda n, r: [sorted_ints(0, 9)(n, r)],
     configs={"with": {}, "without": WITHOUT_FACTS,
              "count": {"ZINNIA_FORCE_CONSTRUCTION": "argextreme: sorted line"},
              "position": {"ZINNIA_FORCE_CONSTRUCTION": "argextreme: sorted line by a hinted position"}})
rule(name="sorted_searchsorted", description="np.searchsorted of 4 queries in a sorted array: hinted positions",
     signature=ARR + ', t: Integer[Array, "4"]', body="return np.searchsorted(a, t), np.searchsorted(a, t, side='right')",
     requires=["lambda a, t: is_sorted(a)"], inputs=lambda n, r: [sorted_ints(0, 99)(n, r), np.array([-1, 30, 30, 100])],
     configs={"with": {}, "without": WITHOUT_FACTS, "count": {"ZINNIA_FORCE_CONSTRUCTION": "searchsorted: count"},
              "position": {"ZINNIA_FORCE_CONSTRUCTION": "searchsorted: hinted position"}})
rule(name="sorted_first_above", description="(a > t).argmax() of a sorted array: a scan of monotone flags",
     signature=ARR + ", t: Integer", body="return (a > t).argmax()", requires=["lambda a, t: is_sorted(a)"],
     inputs=lambda n, r: [sorted_ints(0, 99)(n, r), 50],
     configs={"with": {}, "without": WITHOUT_FACTS,
              "count": {"ZINNIA_FORCE_CONSTRUCTION": "gather: scan of monotone flags by a count"},
              "position": {"ZINNIA_FORCE_CONSTRUCTION": "gather: scan of monotone flags by a hinted position"}})
rule(name="sort_of_sorted", description="np.max(np.sort(a)): a sort of the sorted outputs of a sort",
     signature='a: Float[Array, "{n}"]', body="s = np.sort(a)\nreturn np.max(s), np.median(s)", requires=[],
     inputs=lambda n, r: [r.uniform(-5, 5, n)])

rule(name="sort_counts", description="(np.sort(a) > t[i]).argmax() for 5 queries: positions in the sorted values",
     signature='a: Float[Array, "{n}"], t: Float[Array, "5"]', body="s = np.sort(a)\nreturn [(s > t[i]).argmax() for i in range(5)]",
     requires=[], inputs=lambda n, r: [r.uniform(-5, 5, n), np.array([-6.0, -1.0, 0.0, 2.5, 6.0])],
     configs={"with": {}, "without": {"ZINNIA_FORCE_CONSTRUCTION": "sort: network,sort: pinned selection"},
              "position": {"ZINNIA_FORCE_CONSTRUCTION": "sort: counts by positions in the sorted values"}})

# ── Permutations ────────────────────────────────────────────────────────
rule(name="argsort_scatter", description="c[np.argsort(a)] = v: the argsort is a permutation, a scatter by hits",
     signature='a: Float[Array, "{n}"], v: Integer[Array, "{n}"]', body="c = np.zeros({n}, dtype=int)\nc[np.argsort(a)] = v\nreturn c",
     requires=[], inputs=lambda n, r: [r.uniform(-5, 5, n), ints(0, 9)(n, r)], sizes=(8, 16, 32, 64))
rule(name="permutation_gather", description="v[p] for p checked a permutation: its values need no window or index checks",
     signature='p: Integer[Array, "{n}"], v: Integer[Array, "{n}"]', body="return v[p]",
     requires=["lambda p, v: is_permutation(p)"], inputs=lambda n, r: [r.permutation(n).astype(np.int64), ints(0, 9)(n, r)],
     sizes=(8, 16, 32, 64))

# ── Domain checks ───────────────────────────────────────────────────────
rule(name="sqrt_sum_squares", description="np.sqrt((x * x).sum()): the sum of squares is non-negative by definition",
     signature='x: Float[Array, "{n}"]', body="return np.sqrt((x * x).sum())", requires=[],
     inputs=lambda n, r: [r.uniform(-9, 9, n)])
