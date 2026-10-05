#!/usr/bin/env python3
"""Inputs A and B for the DS-1000 performance-set kernels (`fixture.json`).

Every case has one draw function below. It mirrors the DS-1000
`define_test_input` branch the kernel's shapes come from, with NumPy's legacy
`RandomState` so the code reads like DS-1000's (`rs.rand(20)` for
`np.random.rand(20)`). Where a draw differs from DS-1000, its description
says how and why; the usual reasons are

* a DS-1000 test uses constant data (`np.ones`), which would make every output
  equal and hide a wrong copy, so the draw uses `rand` instead;
* the shape test case is a literal, so the draw uses the distribution of the
  random test case at the literal's shape;
* a value must stay inside the kernel's domain (Q32 range, valid indices).

Seeds: input A uses `RandomState(1000 * problem_id + 1)`, B uses `+ 2`. A draw
may reject samples that do not exercise the kernel's data-dependent paths
(`require`); the same RandomState keeps drawing, and the notes record how many
samples were rejected.

Each fixture is checked here: the kernel body runs as plain NumPy on A and B
without error; every input and output is finite, every float within the Q32
range |v| < 2^31; A and B differ; the outputs differ unless the case says
otherwise; data-movement kernels only output input values or constants. With
`--zinnia`, Zinnia also compiles the kernel and mock-proves both inputs, and
the mock's public values must contain the NumPy outputs.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ds1000_perf as perf  # noqa: E402
import ds1000_records as records  # noqa: E402
import generate_ds1000_kernels as generator  # noqa: E402
import numpy_reference  # noqa: E402

Q32_LIMIT = 2.0 ** 31
INPUT_NAMES = ("A", "B")
MAX_ATTEMPTS = 2000

Values = dict[str, Any]
Require = Callable[[Values, Any], str | None]


@dataclass
class Draw:
    fn: Callable[..., Values]
    test_case: int
    how: str
    require: Require | None = None
    outputs_may_match: bool = False


DRAWS: dict[int, Draw] = {}


def draw(pid: int, test_case: int, how: str, require: Require | None = None,
         outputs_may_match: bool = False) -> Callable[[Callable[..., Values]], Callable[..., Values]]:
    def register(fn: Callable[..., Values]) -> Callable[..., Values]:
        DRAWS[pid] = Draw(fn, test_case, how, require, outputs_may_match)
        return fn
    return register


def seed(pid: int, name: str) -> int:
    return 1000 * pid + 1 + INPUT_NAMES.index(name)


# ---------------------------------------------------------------------------
# Requirements (return a reason when the sample does not exercise the path)
# ---------------------------------------------------------------------------


def interior(index: int, size: int, what: str) -> str | None:
    return None if 0 < index < size - 1 else f"{what} at an end ({index} of {size})"


def interior_argmax(v: Values, out: Any) -> str | None:
    return interior(int(np.argmax(v["a"])), v["a"].size, "argmax")


def interior_argmin(v: Values, out: Any) -> str | None:
    return interior(int(np.argmin(v["a"])), v["a"].size, "argmin")


def partial_selection(count: int, total: int, what: str = "selection") -> str | None:
    return None if 0 < count < total else f"{what} has {count} of {total} elements"


def distinct(*names: str) -> Require:
    def check(v: Values, out: Any) -> str | None:
        for name in names:
            flat = np.asarray(v[name]).ravel()
            if np.unique(flat).size != flat.size:
                return f"{name} has repeated values"
        return None
    return check


def all_of(*checks: Require) -> Require:
    def check(v: Values, out: Any) -> str | None:
        for item in checks:
            reason = item(v, out)
            if reason:
                return reason
        return None
    return check


def by_input(fn: Callable[..., Values]) -> bool:
    return len(inspect.signature(fn).parameters) == 2


# ---------------------------------------------------------------------------
# Draws. Docstring-free on purpose: `how` is the documentation that goes into
# fixture.json, so it must say what DS-1000 does and how this draw differs.
# ---------------------------------------------------------------------------

# One-hot encodings with a data-dependent width (0295-0299).
@draw(295, 2, "DS-1000 test 2: a = randint(0, 20, 50)")
def _(rs): return {"a": rs.randint(0, 20, 50)}


@draw(297, 2, "DS-1000 test 2: a = randint(-5, 20, 50)")
def _(rs): return {"a": rs.randint(-5, 20, 50)}


@draw(298, 3, "shape of DS-1000 test 3 (5 values, 3 of them distinct); values chosen from rand(3) "
               "like test 2's rand; requires repeated values",
      require=lambda v, out: partial_selection(np.unique(v["a"]).size - 1, v["a"].size - 1, "distinct values"))
def _(rs): return {"a": rs.choice(rs.rand(3), 5)}


@draw(299, 2, "DS-1000 test 2: a = randint(0, 20, (10, 20))")
def _(rs): return {"a": rs.randint(0, 20, (10, 20))}


@draw(300, 2, "DS-1000 test 2: a = rand(20), p = randint(1, 99); requires a non-zero interpolation "
               "fraction (p * 19 not a multiple of 100)",
      require=lambda v, out: None if (v["p"] * 19) % 100 else "percentile falls on a sample")
def _(rs): return {"a": rs.rand(20), "p": int(rs.randint(1, 99))}


@draw(301, 2, "DS-1000 test 2: A = rand(20)")
def _(rs): return {"A": rs.rand(20)}


@draw(302, 2, "DS-1000 test 2: A = rand(20)")
def _(rs): return {"A": rs.rand(20)}


@draw(303, 2, "DS-1000 test 2: A = rand(23)")
def _(rs): return {"A": rs.rand(23)}


@draw(304, 2, "DS-1000 test 2: A = rand(23)")
def _(rs): return {"A": rs.rand(23)}


# argmax / argmin family on rand(8, 6) or rand(5, 6): extremum not at either end.
for _pid in (309, 311, 312):
    draw(_pid, 2, "DS-1000 test 2: a = rand(8, 6) (the shape its seeded randint draws); "
                  "requires the maximum away from both ends", interior_argmax)(lambda rs: {"a": rs.rand(8, 6)})
draw(310, 2, "DS-1000 test 2: a = rand(8, 6); requires the minimum away from both ends",
     interior_argmin)(lambda rs: {"a": rs.rand(8, 6)})


def _second_argmax_interior(v: Values, out: Any) -> str | None:
    a = v["a"].copy()
    first = int(np.argmax(a))
    a.flat[first] = a.min()
    return interior(first, a.size, "first argmax") or interior(int(np.argmax(a)), a.size, "second argmax")


draw(314, 2, "DS-1000 test 2: a = rand(8, 6); requires both maxima the kernel finds away from the ends",
     _second_argmax_interior)(lambda rs: {"a": rs.rand(8, 6)})


@draw(317, 2, "DS-1000 test 2: a = rand(10, 20, 5) (DS-1000 passes it as a nested list; the kernel takes the array)")
def _(rs): return {"a": rs.rand(10, 20, 5)}


def _shuffled(rs, n):
    p = np.arange(n)
    rs.shuffle(p)
    return p


@draw(318, 2, "DS-1000 test 2: a = rand(8, 6), permutation = shuffled arange(6); requires a non-identity permutation",
      require=lambda v, out: None if np.any(v["permutation"] != np.arange(6)) else "identity permutation")
def _(rs): return {"a": rs.rand(8, 6), "permutation": _shuffled(rs, 6)}


@draw(319, 2, "DS-1000 test 2: a = rand(8, 6, 8), permutation = shuffled arange(8); requires a non-identity permutation",
      require=lambda v, out: None if np.any(v["permutation"] != np.arange(8)) else "identity permutation")
def _(rs): return {"a": rs.rand(8, 6, 8), "permutation": _shuffled(rs, 8)}


draw(320, 2, "DS-1000 test 2: a = rand(5, 6); requires the minimum away from both ends",
     interior_argmin)(lambda rs: {"a": rs.rand(5, 6)})
draw(321, 2, "DS-1000 test 2: a = rand(5, 6); requires the maximum away from both ends",
     interior_argmax)(lambda rs: {"a": rs.rand(5, 6)})


@draw(322, 3, "shape of DS-1000 test 3 (2x2, two minima); values randint(1, 5) like test 2; requires the "
               "minimum to occur 2 or 3 times (argwhere returns several but not all cells)",
      require=lambda v, out: partial_selection(int(np.sum(v["a"] == v["a"].min())) - 1, 3, "extra minima"))
def _(rs): return {"a": rs.randint(1, 5, (2, 2))}


draw(323, 2, "DS-1000 test 2: degree = randint(0, 360)")(lambda rs: {"degree": int(rs.randint(0, 360))})
draw(324, 2, "DS-1000 test 2: degree = randint(0, 360)")(lambda rs: {"degree": int(rs.randint(0, 360))})


@draw(325, 3, "DS-1000 tests 2-3: number = randint(0, 360); A gives 1 and B gives 0 (both comparison outcomes)",
      require=None)
def _(rs, which):
    while True:
        number = int(rs.randint(0, 360))
        if int(np.sin(number) > np.sin(np.deg2rad(number))) == (1 if which == "A" else 0):
            return {"number": number}


draw(326, 2, "DS-1000 test 2: value = (rand() - 0.5) * 2")(lambda rs: {"value": float((rs.rand() - 0.5) * 2)})
draw(327, 2, "DS-1000 test 2: A = rand(10)")(lambda rs: {"A": rs.rand(10)})
draw(328, 3, "shape of DS-1000 test 3 (5 values); values rand(5) like test 2 (the dtype is Float because test 2 "
              "is float)")(lambda rs: {"A": rs.rand(5)})
draw(329, 2, "DS-1000 test 2: a = rand(8, 6)")(lambda rs: {"a": rs.rand(8, 6)})


@draw(331, 2, "DS-1000 test 2: numerator, denominator = randint(2, 10) each; requires a common factor",
      require=lambda v, out: None if math.gcd(v["numerator"], v["denominator"]) > 1 else "already reduced")
def _(rs): return {"numerator": int(rs.randint(2, 10)), "denominator": int(rs.randint(2, 10))}


draw(334, 2, "DS-1000 test 2: a, b, c = rand(50) each")(
    lambda rs: {"a": rs.rand(50), "b": rs.rand(50), "c": rs.rand(50)})
draw(335, 2, "DS-1000 test 2: a, b, c = rand(50) each")(
    lambda rs: {"a": rs.rand(50), "b": rs.rand(50), "c": rs.rand(50)})
draw(336, 2, "DS-1000 test 2: a = rand(8, 8)")(lambda rs: {"a": rs.rand(8, 8)})
draw(337, 2, "DS-1000 test 2: a = rand(8, 10)")(lambda rs: {"a": rs.rand(8, 10)})
draw(338, 2, "DS-1000 test 2: a = rand(8, 8)")(lambda rs: {"a": rs.rand(8, 8)})
draw(339, 2, "DS-1000 test 2: a = rand(8, 10)")(lambda rs: {"a": rs.rand(8, 10)})
draw(340, 1, "DS-1000 test 1: X = randint(2, 10, (5, 6))")(lambda rs: {"X": rs.randint(2, 10, (5, 6))})
draw(343, 1, "DS-1000 test 1: X = randint(2, 10, (5, 6))")(lambda rs: {"X": rs.randint(2, 10, (5, 6))})


@draw(345, 2, "DS-1000 test 2: a = rand(8, 6), multiply_number = rand(); col drawn from 1..6 (DS-1000 uses the "
               "literals 2 and 4) so A and B can select different columns")
def _(rs): return {"a": rs.rand(8, 6), "col": int(rs.randint(1, 7)), "multiply_number": float(rs.rand())}


@draw(346, 2, "DS-1000 test 2: a = rand(8, 6), multiply_number = rand(); row drawn from 1..8 (DS-1000 uses the "
               "literals 2 and 4)")
def _(rs): return {"a": rs.rand(8, 6), "row": int(rs.randint(1, 9)), "multiply_number": float(rs.rand())}


@draw(347, 2, "DS-1000 test 2: a = rand(8, 6), divide_number = rand() + 1; row drawn from 1..8 (DS-1000 uses the "
               "literals 2 and 4)")
def _(rs): return {"a": rs.rand(8, 6), "row": int(rs.randint(1, 9)), "divide_number": float(rs.rand() + 1)}


def _rows_in(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.array([any(np.array_equal(row, other) for other in B) for row in A])


@draw(353, 3, "shapes of DS-1000 test 3 (A 2x3, B 7x3); entries randint(0, 2) like test 2; requires exactly one "
               "row of A to occur in B",
      require=lambda v, out: partial_selection(int(_rows_in(v["A"], v["B"]).sum()), 2, "rows of A in B"))
def _(rs): return {"A": rs.randint(0, 2, (2, 3)), "B": rs.randint(0, 2, (7, 3))}


@draw(354, 3, "shapes of DS-1000 test 3; entries randint(0, 2) like test 2; requires rows of A missing from B and "
               "rows of B missing from A",
      require=lambda v, out: (partial_selection(int(_rows_in(v["A"], v["B"]).sum()), 2, "rows of A in B")
                              or partial_selection(int(_rows_in(v["B"], v["A"]).sum()), 7, "rows of B in A")))
def _(rs): return {"A": rs.randint(0, 2, (2, 3)), "B": rs.randint(0, 2, (7, 3))}


for _pid in (355, 357, 358):
    draw(_pid, 1, "DS-1000 test 1: a = rand(3, 3, 3), b = arange(27).reshape(3, 3, 3)")(
        lambda rs: {"a": rs.rand(3, 3, 3), "b": np.arange(27).reshape(3, 3, 3)})

draw(359, 2, "shape of DS-1000 test 2 (np.ones((3, 3))); values rand(3, 3) so a wrong copy is visible")(
    lambda rs: {"a": rs.rand(3, 3)})
draw(360, 2, "shape of DS-1000 test 2 (np.ones((4, 4))); values rand(4, 4) so a wrong copy is visible")(
    lambda rs: {"a": rs.rand(4, 4)})
draw(361, 2, "shape of DS-1000 test 2 (np.ones((6, 6))); values rand(6, 6) so a wrong copy is visible")(
    lambda rs: {"a": rs.rand(6, 6)})
draw(362, 2, "DS-1000 test 2: a = rand(8, 6)")(lambda rs: {"a": rs.rand(8, 6)})


@draw(363, 2, "DS-1000 test 2: a = rand(100), pos = randint(0, 99), element = 66")
def _(rs): return {"a": rs.rand(100), "pos": int(rs.randint(0, 99)), "element": 66}


@draw(364, 2, "DS-1000 test 2: a = rand(100, 10), pos = randint(0, 99), element = rand(10)")
def _(rs): return {"a": rs.rand(100, 10), "pos": int(rs.randint(0, 99)), "element": rs.rand(10)}


@draw(366, 2, "DS-1000 test 2: a = rand(100, 10), pos = sorted(randint(0, 99, (5,))), element = rand(5, 10)")
def _(rs): return {"a": rs.rand(100, 10), "pos": np.sort(rs.randint(0, 99, 5)), "element": rs.rand(5, 10)}


def _near_copy(rs, row: np.ndarray) -> np.ndarray:
    """`row` with relative changes of about 1e-7: isclose (rtol 1e-5) but not ==."""
    return row * (1 + 1e-7 * rs.choice([-1.0, 1.0], row.shape))


@draw(368, 5, "shape of DS-1000 test 5 (3x3); values rand like test 2. A: rows 1-2 equal row 0 up to relative "
               "changes of 1e-7 (isclose holds, == does not); B: independent rows (isclose fails)")
def _(rs, which):
    if which == "A":
        row = rs.rand(3)
        return {"a": np.stack([row, _near_copy(rs, row), _near_copy(rs, row)])}
    return {"a": rs.rand(3, 3)}


@draw(369, 5, "shape of DS-1000 test 5 (3x3); values rand like test 2. A: every row constant up to relative "
               "changes of 1e-7; B: independent values")
def _(rs, which):
    if which == "A":
        col = rs.rand(3, 1)
        return {"a": np.hstack([col, _near_copy(rs, col), _near_copy(rs, col)])}
    return {"a": rs.rand(3, 3)}


@draw(374, 2, "DS-1000 test 2: grades = (rand(50) - 0.5) * 100, eval = randint(10, 90, (5,))")
def _(rs): return {"grades": (rs.rand(50) - 0.5) * 100, "eval": rs.randint(10, 90, 5)}


@draw(375, 2, "DS-1000 test 2: grades = (rand(50) - 0.5) * 100; threshold drawn uniformly from (0.05, 0.95) "
               "(DS-1000 uses the literals 0.5 and 0.6)")
def _(rs): return {"grades": (rs.rand(50) - 0.5) * 100, "threshold": float(rs.uniform(0.05, 0.95))}


for _pid in (381, 382, 383):
    draw(_pid, 2, "DS-1000 test 2: a = (rand(100) - 0.5) * 100; requires distinct values (ties make argsort's "
                  "order implementation-defined)", distinct("a"))(lambda rs: {"a": (rs.rand(100) - 0.5) * 100})

for _pid in (385, 387, 388, 390):
    draw(_pid, 2, "DS-1000 test 2: a = rand(100, 200)")(lambda rs: {"a": rs.rand(100, 200)})
draw(389, 2, "DS-1000 test 2: a = rand(6, 3, 4)")(lambda rs: {"a": rs.rand(6, 3, 4)})
for _pid in (391, 392, 393):
    draw(_pid, 2, "DS-1000 test 2: a = rand(20, 10)")(lambda rs: {"a": rs.rand(20, 10)})


@draw(398, 2, "DS-1000 test 2: A = randn(30), a = randint(2, 10); b drawn from {-1, 1} instead of randint(2, 10): "
               "with |b| >= 2 the recurrence B[k] = a*A[k] + b*B[k-1] exceeds the Q32 range (2^31) within 30 steps")
def _(rs): return {"A": rs.randn(30), "a": int(rs.randint(2, 10)), "b": int(rs.choice([-1, 1]))}


@draw(399, 2, "DS-1000 test 2: A = randn(30), a = randint(2, 10); b, c drawn from {-1, 1} instead of "
               "randint(2, 10) to keep B within the Q32 range (with |b|, |c| <= 1 |B| grows like Fibonacci)")
def _(rs): return {"A": rs.randn(30), "a": int(rs.randint(2, 10)), "b": int(rs.choice([-1, 1])),
                   "c": int(rs.choice([-1, 1]))}


for _pid in (402, 403):
    draw(_pid, 2, "DS-1000 test 2: a = rand(8, 9, 8, 8, 8) (unused by the solution), index = randint(0, 7, (5,))")(
        lambda rs: {"a": rs.rand(8, 9, 8, 8, 8), "index": rs.randint(0, 7, 5)})

draw(405, 2, "DS-1000 test 2: accmap = randint(0, 5, (100,)), a = randint(-100, 100, (100,))")(
    lambda rs: {"accmap": rs.randint(0, 5, 100), "a": rs.randint(-100, 100, 100)})
draw(406, 2, "DS-1000 test 2: index = randint(0, 5, (100,)), a = randint(-100, 100, (100,))")(
    lambda rs: {"index": rs.randint(0, 5, 100), "a": rs.randint(-100, 100, 100)})
draw(407, 2, "DS-1000 test 2: accmap = randint(-2, 5, (100,)), a = randint(-100, 100, (100,))")(
    lambda rs: {"accmap": rs.randint(-2, 5, 100), "a": rs.randint(-100, 100, 100)})
draw(408, 2, "DS-1000 test 2: index = randint(-2, 5, (100,)), a = randint(-100, 100, (100,))")(
    lambda rs: {"index": rs.randint(-2, 5, 100), "a": rs.randint(-100, 100, 100)})
draw(409, 2, "DS-1000 test 2: x, y = rand(8, 6) each")(lambda rs: {"x": rs.rand(8, 6), "y": rs.rand(8, 6)})
draw(411, 3, "shape of DS-1000 test 3 (np.ones((5, 5))); values rand(5, 5) so a wrong copy is visible")(
    lambda rs: {"a": rs.rand(5, 5)})
draw(412, 2, "DS-1000 test 2: x = rand(10) - 0.5; requires both signs",
     lambda v, out: partial_selection(int(np.sum(v["x"] >= 0)), 10, "non-negative values"))(
    lambda rs: {"x": rs.rand(10) - 0.5})


@draw(413, 2, "DS-1000 test 2: x = (rand(10) - 0.5) as complex with two entries set to -1.1+2j at positions 2 "
               "and 5; here two to four drawn positions get rand() - 0.5 + (rand() + 0.5)j, so the selected values "
               "differ between A and B")
def _(rs):
    x = (rs.rand(10) - 0.5).astype(np.complex128)
    cells = rs.choice(10, rs.randint(2, 5), replace=False)
    x[cells] = (rs.rand(cells.size) - 0.5) + 1j * (rs.rand(cells.size) + 0.5)
    return {"x": x}


for _pid in (414, 415, 417):
    draw(_pid, 2, "DS-1000 test 2: data = rand(8) (the length its seeded randint draws)")(
        lambda rs: {"data": rs.rand(8)})
for _pid in (416, 418, 419):
    draw(_pid, 2, "DS-1000 test 2: data = rand(8, 6)")(lambda rs: {"data": rs.rand(8, 6)})


@draw(420, 3, "DS-1000 tests use x in {0.25, -1, 2} with x_min = 0, x_max = 1. A: x uniform in (0, 1) (smooth "
               "branch); B: x uniform in (1, 2) (clamped branch); x_min = 0, x_max = 1 as in DS-1000")
def _(rs, which):
    low, high = (0.0, 1.0) if which == "A" else (1.0, 2.0)
    return {"x": float(rs.uniform(low, high)), "x_min": 0, "x_max": 1}


draw(422, 2, "DS-1000 test 2: a, b = rand(50) each")(lambda rs: {"a": rs.rand(50), "b": rs.rand(50)})
draw(425, 2, "DS-1000 test 2: a = randint(0, 100, (20,))")(lambda rs: {"a": rs.randint(0, 100, 20)})
draw(426, 3, "DS-1000 test 3: a = randint(-1000, 1000, (20,))")(lambda rs: {"a": rs.randint(-1000, 1000, 20)})
draw(427, 2, "DS-1000 test 2: a = randint(0, 100, (20,))")(lambda rs: {"a": rs.randint(0, 100, 20)})
draw(428, 2, "DS-1000 test 2: a = randn(30)")(lambda rs: {"a": rs.randn(30)})
draw(429, 2, "DS-1000 test 2: a = randn(30)")(lambda rs: {"a": rs.randn(30)})


def _outliers(v: Values, out: Any) -> str | None:
    a = v["a"]
    low, high = a.mean() - 2 * a.std(), a.mean() + 2 * a.std()
    return partial_selection(int(np.sum((a <= low) | (a >= high))), a.size, "outliers")


draw(431, 3, "shape of DS-1000 test 3 (8 values, one outlier); values randn(8) like test 2; requires at least one "
              "value outside mean +- 2 std", _outliers)(lambda rs: {"a": rs.randn(8)})


@draw(433, 1, "DS-1000 test 1 is a 4x4 literal with entries 0..3 and zero_rows = zero_cols = 0; here a = "
               "randint(0, 4, (4, 4)) and zero_rows, zero_cols = randint(0, 4) each")
def _(rs): return {"a": rs.randint(0, 4, (4, 4)), "zero_rows": int(rs.randint(0, 4)), "zero_cols": int(rs.randint(0, 4))}


@draw(434, 1, "DS-1000 test 1 is a 4x4 literal with zero_rows = [1, 3], zero_cols = [1, 2]; here a = "
               "randint(0, 4, (4, 4)) and two distinct indices each")
def _(rs): return {"a": rs.randint(0, 4, (4, 4)), "zero_rows": np.sort(rs.choice(4, 2, replace=False)),
                   "zero_cols": np.sort(rs.choice(4, 2, replace=False))}


draw(435, 2, "DS-1000 tests are 4x4 literals with entries 0..5; here a = randint(0, 6, (4, 4))")(
    lambda rs: {"a": rs.randint(0, 6, (4, 4))})
draw(436, 2, "DS-1000 test 2: a = rand(10, 5)")(lambda rs: {"a": rs.rand(10, 5)})
draw(437, 2, "DS-1000 test 2: a = rand(10, 5)")(lambda rs: {"a": rs.rand(10, 5)})
draw(438, 1, "DS-1000 test 1 lists post = [2, 5, 6, 10], distance = [50, 100, 500, 1000]; here post = "
              "randint(1, 20, 4), distance = randint(10, 1000, 4), each with distinct values",
     distinct("post", "distance"))(lambda rs: {"post": rs.randint(1, 20, 4), "distance": rs.randint(10, 1000, 4)})
draw(439, 2, "DS-1000 test 2: X = rand(10, 20)")(lambda rs: {"X": rs.rand(10, 20)})
draw(440, 2, "DS-1000 test 2: X = rand(10, 5, 5) (bound to Y by the exec_context)")(
    lambda rs: {"Y": rs.rand(10, 5, 5)})


@draw(441, 2, "DS-1000 tests are literals (a of length 4, number in or not in a); here a = randint(0, 10, 4) and "
               "number = randint(0, 10). A: number is in a; B: it is not")
def _(rs, which):
    while True:
        a, number = rs.randint(0, 10, 4), int(rs.randint(0, 10))
        if (number in a) == (which == "A"):
            return {"a": a, "number": number}


for _pid in (442, 443):
    draw(_pid, 2, "DS-1000 test 2: A = randint(0, 10, (20,)), B = randint(0, 10, (3,)); requires some but not all "
                  "of A in B", lambda v, out: partial_selection(int(np.isin(v["A"], v["B"]).sum()), 20, "A in B"))(
        lambda rs: {"A": rs.randint(0, 10, 20), "B": rs.randint(0, 10, 3)})


@draw(444, 3, "DS-1000 test 3: A = randint(0, 10, (20,)); B is the literal [2, 3, 5] there, here 3 sorted distinct "
               "values from 0..9; requires a selection that is neither empty nor all of A",
      require=lambda v, out: partial_selection(int(np.asarray(out).size), 20))
def _(rs): return {"A": rs.randint(0, 10, 20), "B": np.sort(rs.choice(10, 3, replace=False))}


draw(448, 2, "DS-1000 test 2: x_dists, y_dists = rand(3, 4) each")(
    lambda rs: {"x_dists": rs.rand(3, 4), "y_dists": rs.rand(3, 4)})


@draw(450, 2, "DS-1000 test 2: a = rand(7, 8, 9); second, third are the literals [0, 4], [6, 7] there, here two "
               "distinct in-range indices each")
def _(rs): return {"a": rs.rand(7, 8, 9), "second": rs.choice(8, 2, replace=False),
                   "third": rs.choice(9, 2, replace=False)}


for _pid in (452, 453, 454):
    draw(_pid, 2, "DS-1000 tests are 5x4 integer literals with entries in -25..25 and mixed signs; here "
                  "randint(-25, 26, (5, 4)); requires no all-zero row (the row norm divides)",
         lambda v, out: None if np.all(np.abs(v["X"]).sum(axis=1) > 0) else "zero row")(
        lambda rs: {"X": rs.randint(-25, 26, (5, 4))})

draw(462, 2, "DS-1000 test 2: a = randint(0, 3, (20,))")(lambda rs: {"a": rs.randint(0, 3, 20)})
draw(463, 2, "DS-1000 test 2: a = randint(0, 3, (20,)).reshape(-1, 1)")(
    lambda rs: {"a": rs.randint(0, 3, 20).reshape(-1, 1)})
for _pid in (467, 468):
    draw(_pid, 1, "DS-1000 test 1 is the 4x4 literal a[i][j] = i + j + 1; here a = randint(1, 10, (4, 4))")(
        lambda rs: {"a": rs.randint(1, 10, (4, 4))})
draw(471, 2, "DS-1000 test 2: a = rand(6, 2, 9, 3, 7, 4) (the shape its seeded randint draws; bound to Z)")(
    lambda rs: {"Z": rs.rand(6, 2, 9, 3, 7, 4)})
draw(472, 2, "DS-1000 test 2: a = rand(6, 2, 9, 3, 7, 4)")(lambda rs: {"a": rs.rand(6, 2, 9, 3, 7, 4)})
draw(478, 1, "DS-1000 test 1 is a 4x4 integer literal with entries 0..10; here randint(0, 11, (4, 4))")(
    lambda rs: {"a": rs.randint(0, 11, (4, 4))})


@draw(480, 2, "DS-1000 test 2: x, y = randint(2, 7, (8,)), a, b = randint(2, 7). A: some index matches (x, y) == "
               "(a, b), not index 0; B: no index matches (result -1)")
def _(rs, which):
    while True:
        v = {"x": rs.randint(2, 7, 8), "y": rs.randint(2, 7, 8), "a": int(rs.randint(2, 7)), "b": int(rs.randint(2, 7))}
        hits = np.flatnonzero((v["x"] == v["a"]) & (v["y"] == v["b"]))
        if (which == "A" and hits.size and hits[0] > 0) or (which == "B" and not hits.size):
            return v


@draw(481, 3, "shape of DS-1000 test 3 (9 values); x, y = randint(2, 7, (9,)), a, b = randint(2, 7) like test 2; "
               "requires at least one match",
      require=lambda v, out: partial_selection(int(np.asarray(out).size), 9, "matches"))
def _(rs): return {"x": rs.randint(2, 7, 9), "y": rs.randint(2, 7, 9), "a": int(rs.randint(2, 7)), "b": int(rs.randint(2, 7))}


draw(485, 1, "DS-1000 test 1: A = rand(5, 6, 3), B = rand(3, 3)")(lambda rs: {"A": rs.rand(5, 6, 3), "B": rs.rand(3, 3)})
draw(489, 1, "DS-1000 test 1: a = (rand(100, 50) - 0.5) * 50 (bound to arr)")(
    lambda rs: {"arr": (rs.rand(100, 50) - 0.5) * 50})


@draw(490, 1, "DS-1000 test 1: arr = (rand(5, 50) - 0.5) * 50; n1, n2 are the literals [1, 2, 3, 4, -5], "
               "[6, 7, 8, 9, 10] there, here n1 = randint(-5, 5, 5) and n2 = n1 + randint(1, 10, 5)")
def _(rs):
    n1 = rs.randint(-5, 5, 5)
    return {"arr": (rs.rand(5, 50) - 0.5) * 50, "n1": n1, "n2": n1 + rs.randint(1, 10, 5)}


@draw(491, 2, "DS-1000 test 2: s1, s2 = rand(6, 1) each; then 2 to 4 rows of s2 are set to the s1 value times "
               "(1 + 1e-7) (test 1's near-equal sums), so some pairs are close and some are not",
      require=lambda v, out: partial_selection(int(out), 6, "different pairs"))
def _(rs):
    s1, s2 = rs.rand(6, 1), rs.rand(6, 1)
    rows = rs.choice(6, rs.randint(2, 5), replace=False)
    s2[rows] = s1[rows] * (1 + 1e-7)
    return {"s1": s1, "s2": s2}


draw(495, 1, "shape of DS-1000 test 1 (np.ones((41, 13))); values rand so a wrong copy is visible")(
    lambda rs: {"a": rs.rand(41, 13)})
draw(496, 4, "shape of DS-1000 test 4 (np.ones((42, 10))); values rand so a wrong copy is visible")(
    lambda rs: {"a": rs.rand(42, 10)})
draw(497, 4, "shape of DS-1000 test 4 (np.ones((42, 10))); values rand; element drawn from 1..9 (DS-1000 uses 5)")(
    lambda rs: {"a": rs.rand(42, 10), "element": int(rs.randint(1, 10))})
draw(499, 4, "shape of DS-1000 test 4 (np.ones((42, 10))); values rand so a wrong copy is visible")(
    lambda rs: {"a": rs.rand(42, 10)})
draw(500, 2, "DS-1000 test 2: a = rand(3 * 18) (n = 18 from its seeded randint)")(lambda rs: {"a": rs.rand(54)})


@draw(501, 2, "DS-1000 test 2: a = rand(13, 13, 2), b = zeros with five ones; the five positions are drawn "
               "(DS-1000 fixes them on a diagonal band)")
def _(rs):
    b = np.zeros((13, 13), dtype=int)
    cells = rs.choice(169, 5, replace=False)
    b.flat[cells] = 1
    return {"a": rs.rand(13, 13, 2), "b": b}


def _ds1000_index_planes(rs, dim: int, depth: int) -> np.ndarray:
    b = np.zeros((dim, dim), dtype=int)
    for i in range(depth):
        row, col = rs.randint(0, dim - 1, 5), rs.randint(0, dim - 1, 5)
        b[row, col] = i
    return b


for _pid in (503, 504, 505):
    draw(_pid, 2, "DS-1000 test 2 with dim = 13, T = 5: a = rand(13, 13, 5); b = zeros, then for i in range(5) "
                  "five random cells set to i")(
        lambda rs: {"a": rs.rand(13, 13, 5), "b": _ds1000_index_planes(rs, 13, 5)})


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    if not rows.size:
        return None
    return rows.min(), rows.max(), cols.min(), cols.max()


def _proper_bbox(mask: np.ndarray) -> str | None:
    box = _bbox(mask)
    if box is None:
        return "empty bounding box"
    if box == (0, mask.shape[0] - 1, 0, mask.shape[1] - 1):
        return "bounding box is the whole array"
    return None


@draw(507, 2, "DS-1000 test 2: im = randint(0, 2, (5, 6)), im[:, 0] = 0, im[-1, :] = 0; requires a non-empty crop",
      require=lambda v, out: _proper_bbox(v["im"] != 0))
def _(rs):
    im = rs.randint(0, 2, (5, 6))
    im[:, 0] = 0
    im[-1, :] = 0
    return {"im": im}


@draw(508, 2, "DS-1000 test 2 is randint(0, 2, (10, 10)), whose bounding box is almost always the whole array; "
               "here zeros with randint(0, 2) values in a random sub-rectangle (like test 1's sparse literal); "
               "requires a bounding box smaller than the array",
      require=lambda v, out: _proper_bbox(v["A"] != 0))
def _(rs):
    A = np.zeros((10, 10), dtype=int)
    r0, c0 = rs.randint(0, 5, 2)
    r1, c1 = r0 + rs.randint(2, 6), c0 + rs.randint(2, 6)
    A[r0:r1, c0:c1] = rs.randint(0, 2, (r1 - r0, c1 - c0))
    return {"A": A}


@draw(509, 2, "DS-1000 test 2 at the kernel's 10x10 shape: im = randint(0, 10, (10, 10)) as floats, im[:, 0] = 5, "
               "im[-1, :] = 5; requires zeros whose bounding box is smaller than the array",
      require=lambda v, out: _proper_bbox(v["im"] == 0))
def _(rs):
    im = rs.randint(0, 10, (10, 10)).astype(float)
    im[:, 0] = 5
    im[-1, :] = 5
    return {"im": im}


@draw(510, 2, "DS-1000 test 2 at the kernel's 10x10 shape: im = randint(0, 10, (10, 10)) as floats, im[:, 0] = 0, "
               "im[-1, :] = 0; requires a bounding box smaller than the array",
      require=lambda v, out: _proper_bbox(v["im"] != 0))
def _(rs):
    im = rs.randint(0, 10, (10, 10)).astype(float)
    im[:, 0] = 0
    im[-1, :] = 0
    return {"im": im}


# ---------------------------------------------------------------------------
# Fixture construction and checks
# ---------------------------------------------------------------------------

JSON_DTYPE = {"Float": "float", "Integer": "int", "Boolean": "bool", "Complex": "complex"}


class FixtureError(Exception):
    pass


def to_json_value(value: Any, dtype: str) -> Any:
    array = np.asarray(value)
    if dtype == "Complex":
        array = np.stack([array.real, array.imag], axis=-1).astype(float)
        return array.tolist()
    if dtype == "Float":
        return array.astype(float).tolist()
    if dtype == "Boolean":
        return array.astype(bool).tolist()
    return array.astype(np.int64).tolist()


def flatten_numbers(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return [x for item in value for x in flatten_numbers(item)]
    return list(np.asarray(value).ravel())


def check_values(kernel: generator.GeneratedKernel, values: Values) -> None:
    missing = [n for n, _ in kernel.params if n not in values]
    extra = [n for n in values if n not in dict(kernel.params)]
    if missing or extra:
        raise FixtureError(f"draw returns wrong names (missing {missing}, extra {extra})")
    for name, ptype in kernel.params:
        array = np.asarray(values[name])
        shape = () if ptype.shape is None else ptype.shape
        if array.shape != shape:
            raise FixtureError(f"{name}: shape {array.shape}, kernel expects {shape}")
        if ptype.dtype == "Integer" and array.dtype.kind not in "iu":
            raise FixtureError(f"{name}: dtype {array.dtype}, kernel expects Integer")
        if ptype.dtype in ("Float", "Complex") and not np.all(np.isfinite(array)):
            raise FixtureError(f"{name}: non-finite value")
        if ptype.dtype == "Float" and np.any(np.abs(array) >= Q32_LIMIT):
            raise FixtureError(f"{name}: value outside the Q32 range")


def check_output(output: Any) -> None:
    for x in flatten_numbers(output):
        if isinstance(x, (np.complexfloating, complex)):
            x = abs(x)
        if isinstance(x, (float, np.floating)) and (not math.isfinite(x) or abs(x) >= Q32_LIMIT):
            raise FixtureError(f"output value {x} is not finite or outside the Q32 range")


def same_output(a: Any, b: Any) -> bool:
    fa, fb = flatten_numbers(a), flatten_numbers(b)
    return len(fa) == len(fb) and all(np.asarray(x) == np.asarray(y) for x, y in zip(fa, fb))


def data_movement_violation(values: Values, output: Any, constants: set[float]) -> str | None:
    pool = {complex(x) for v in values.values() for x in np.asarray(v).ravel()} | {complex(c) for c in constants}
    for x in flatten_numbers(output):
        if complex(x) not in pool:
            return f"output value {x} is neither an input value nor a constant"
    return None


@dataclass
class Fixture:
    data: dict[str, Any]
    outputs: dict[str, Any]


def build_fixture(case_dir: Path, record: dict[str, Any]) -> Fixture:
    pid = record["problem_id"]
    spec = DRAWS[pid]
    kernel = perf.expected_kernel(record)
    kernel_path = case_dir / "kernel.py"
    inputs: dict[str, Values] = {}
    outputs: dict[str, Any] = {}
    notes: list[str] = []
    for name in INPUT_NAMES:
        rs = np.random.RandomState(seed(pid, name))
        for attempt in range(MAX_ATTEMPTS):
            values = spec.fn(rs, name) if by_input(spec.fn) else spec.fn(rs)
            check_values(kernel, values)
            output = numpy_reference.run_kernel(kernel_path, [values[n] for n, _ in kernel.params])
            reason = spec.require(values, output) if spec.require else None
            if reason is None:
                break
        else:
            raise FixtureError(f"input {name}: no sample in {MAX_ATTEMPTS} attempts exercises the paths ({reason})")
        check_output(output)
        if pid in perf.DATA_MOVEMENT_ONLY:
            constants = {0.0} | {float(np.asarray(v).ravel()[0]) for v in values.values() if np.asarray(v).ndim == 0}
            violation = data_movement_violation(values, output, constants)
            if violation:
                raise FixtureError(f"input {name}: listed as data movement, but {violation}")
        if attempt:
            notes.append(f"input {name}: {attempt} sample(s) rejected before one exercised the paths")
        inputs[name] = values
        outputs[name] = output
    if all(np.array_equal(np.asarray(inputs["A"][n]), np.asarray(inputs["B"][n])) for n, _ in kernel.params):
        raise FixtureError("inputs A and B are equal")
    if same_output(outputs["A"], outputs["B"]) and not spec.outputs_may_match:
        raise FixtureError("outputs on A and B are equal (degenerate kernel or uninformative inputs?)")

    parameters = []
    for name, ptype in kernel.params:
        entry: dict[str, Any] = {"name": name, "kind": "scalar" if ptype.shape is None else "array",
                                 "dtype": JSON_DTYPE[ptype.dtype]}
        if ptype.shape is not None:
            entry["shape"] = list(ptype.shape)
        parameters.append(entry)
    data = {
        "case": case_dir.name,
        "function": "kernel",
        "parameters": parameters,
        "constants": {name: generator.plain_value(np.asarray(v).tolist()) for name, v in kernel.constants.items()},
        "inputs": {name: {n: to_json_value(inputs[name][n], ptype.dtype) for n, ptype in kernel.params}
                   for name in INPUT_NAMES},
        "notes": " ".join([
            f"Seeded draws: input A uses numpy RandomState({seed(pid, 'A')}), B uses RandomState({seed(pid, 'B')}).",
            f"Draw (DS-1000 test case {spec.test_case}): {spec.how}.",
            *([f"Compile-time constants (not inputs): "
               f"{', '.join(f'{k} = {v}' for k, v in kernel.constants.items())}."] if kernel.constants else []),
            *(["Complex values are [real, imag] pairs."] if any(t.dtype == 'Complex' for _, t in kernel.params) else []),
            *notes,
        ]),
    }
    return Fixture(data, outputs)


def format_fixture(data: dict[str, Any]) -> str:
    """JSON with one line per parameter description and per input value (arrays stay on one line)."""
    compact = {"separators": (", ", ": ")}
    lines = ["{"]
    for i, (key, value) in enumerate(data.items()):
        end = "," if i < len(data) - 1 else ""
        if key == "parameters":
            items = [f"  {json.dumps(p, **compact)}" for p in value]
            lines.append(f' "{key}": [\n' + ",\n".join(items) + f"\n ]{end}")
        elif key == "inputs":
            blocks = []
            for name, values in value.items():
                rows = [f'   "{n}": {json.dumps(v, **compact)}' for n, v in values.items()]
                blocks.append(f'  "{name}": {{\n' + ",\n".join(rows) + "\n  }")
            lines.append(f' "{key}": {{\n' + ",\n".join(blocks) + f"\n }}{end}")
        else:
            lines.append(f' "{key}": {json.dumps(value, **compact)}{end}')
    lines.append("}")
    return "\n".join(lines) + "\n"


def json_output(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [json_output(v) for v in value]
    array = np.asarray(value)
    if array.dtype.kind == "c":
        return np.stack([array.real, array.imag], axis=-1).tolist()
    return array.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cases", help="comma-separated problem ids (default: every case with a draw)")
    parser.add_argument("--write", action="store_true", help="write fixture.json into the case folders")
    parser.add_argument("--zinnia", action="store_true", help="also compile and mock-prove with Zinnia")
    parser.add_argument("--report", type=Path, help="write a JSON verification report here")
    args = parser.parse_args()

    import zinnia_probe

    manifest = records.load_manifest()
    wanted = {int(c) for c in args.cases.split(",")} if args.cases else set(DRAWS)
    report: dict[str, Any] = {}
    for case_dir in perf.case_dirs():
        pid = perf.problem_id(case_dir)
        if pid not in wanted:
            continue
        entry: dict[str, Any] = {}
        try:
            fixture = build_fixture(case_dir, manifest[pid])
            entry["numpy"] = "ok"
            entry["outputs"] = {k: json_output(v) for k, v in fixture.outputs.items()}
            if args.write:
                (case_dir / "fixture.json").write_text(format_fixture(fixture.data), encoding="utf-8")
            if args.zinnia:
                with tempfile.TemporaryDirectory() as scratch:
                    path = Path(scratch) / "fixture.json"
                    path.write_text(json.dumps(fixture.data), encoding="utf-8")
                    for name in INPUT_NAMES:
                        entry[f"zinnia_{name}"] = zinnia_probe.run(case_dir / "kernel.py", path, name)
        except (FixtureError, Exception) as exc:  # noqa: BLE001 - report every failure
            entry["error"] = f"{type(exc).__name__}: {exc}"
        report[case_dir.name] = entry
        print(case_dir.name, entry.get("error", "ok"), flush=True)
    if args.report:
        args.report.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
