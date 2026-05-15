#!/usr/bin/env python3
"""Phase 3a — minimal viable differential fuzzer for Zinnia.

Generates small numpy-like programs from nine shapes:

    A) scalar reduction over a 1-D integer array
    B) elementwise transcendental + scalar reduction over a 1-D float array
    C) matrix multiplication
    D) indexing (scalar / slice / fancy / np.take)
    E) shape-preserving op chains (transpose/reshape/swapaxes/flatten/
       squeeze/expand_dims) → reduce
    F) np.where
    G) concat / stack / vstack / hstack → reduce
    H) deeper compositions (3-5 ops mixing the above)
    I) programs with `@requires` Cmp preconditions on scalar inputs;
       both satisfying and violating inputs are generated. The
       generator emits an `expected_satisfied` flag; for satisfying
       inputs that equals True (standard differential check); for
       violating inputs the witness-check should fire and
       `satisfied=False` is the expected outcome.

For each generated program:

  1. numpy is the oracle — run it directly on random inputs to get the
     reference output value(s).
  2. Zinnia is the device-under-test — compile the same source as a
     `@zk_circuit`, then run it on the same inputs with a synthesized
     `assert <expr> ~= <ref>` tail. The `@zk_circuit` machinery returns
     a ``ZKExecResult`` whose ``.satisfied`` is True iff every assert
     inside the circuit held; that's the signal we compare against.

A divergence is `satisfied == False` on a program whose reference numpy
output is, by construction, the value the assertion was synthesized
against. A compile failure is any exception (incl. BaseException to
catch PyO3 panics) raised while compiling/running Zinnia.

Run:

    cd zinnia-src
    python3.13 tools/fuzz_differential.py --iterations 30

Each non-success iteration writes a JSON report under
``tools/fuzz_reports/<timestamp>/<index>.json``. Success cases are
counted but not written, to keep the report directory small.

Stdlib + numpy + zinnia only.
"""
from __future__ import annotations

import argparse
import json
import linecache
import math
import multiprocessing as mp
import os
import random
import sys
import time
import traceback
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parent.parent
REPORT_ROOT = REPO_SRC / "tools" / "fuzz_reports"

# Make sure `import zinnia` works regardless of where the script was launched
# from (the impl card says to invoke from the repo root). zinnia-src is the
# package's parent directory.
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

import numpy as _numpy  # noqa: E402 — must happen after sys.path tweak above.

SHAPE_SIZES = [2, 4, 8]
MATMUL_SIZES = [2, 4]
REDUCTIONS_INT = ["sum", "max", "min"]      # `mean` over ints is float in numpy; skip for shape A
REDUCTIONS_FLOAT = ["sum", "mean", "max", "min"]
ELEMENTWISE_FLOAT = ["sqrt", "exp", "log", "sin", "cos"]
INT_LO, INT_HI = -10, 10
FLOAT_LO, FLOAT_HI = -3.0, 3.0
FLOAT_TOL = 1e-3   # ZK fixed-point quantisation is ~2^-precision_bits; transcendentals
                    # compound rounding, so 1e-3 is the practical absolute tolerance for
                    # the differential signal. Tighter would yield spurious divergences.
INT_TOL = 0         # ints are exact
TIMEOUT_SEC = 30


# ----------------------------------------------------------------------------
# Generator
# ----------------------------------------------------------------------------

def _gen_shape_a(rng: random.Random) -> tuple[str, dict, str, object, bool]:
    """Shape A: 1-D integer reduction."""
    size = rng.choice(SHAPE_SIZES)
    op = rng.choice(REDUCTIONS_INT)
    x = _numpy.array([rng.randint(INT_LO, INT_HI) for _ in range(size)], dtype=_numpy.int64)
    ref = getattr(_numpy, op)(x)
    ref_val = int(ref)
    source = (
        "from zinnia import zk_circuit\n"
        "from zinnia.lang.operator import NDArray, Integer, Float\n"
        "import numpy as np\n"
        "\n"
        "@zk_circuit\n"
        f"def f(x: NDArray[Integer, {size}]):\n"
        f"    out = np.{op}(x)\n"
        f"    assert out == {ref_val}\n"
    )
    return source, {"x": x.tolist()}, "A", ref_val, True


def _gen_shape_b(rng: random.Random) -> tuple[str, dict, str, object, bool]:
    """Shape B: elementwise float + reduce."""
    size = rng.choice(SHAPE_SIZES)
    elem_op = rng.choice(ELEMENTWISE_FLOAT)
    red_op = rng.choice(REDUCTIONS_FLOAT)
    # `x*x + 1.0` is always > 0, so sqrt/log are safe; exp can blow up so cap x.
    x = _numpy.array(
        [rng.uniform(FLOAT_LO, FLOAT_HI) for _ in range(size)],
        dtype=_numpy.float64,
    )
    y = getattr(_numpy, elem_op)(x * x + 1.0)
    ref = float(getattr(_numpy, red_op)(y))
    # Use a unique-looking variable name for the literal to make minimisation easier later.
    source = (
        "from zinnia import zk_circuit\n"
        "from zinnia.lang.operator import NDArray, Integer, Float\n"
        "import numpy as np\n"
        "\n"
        "@zk_circuit\n"
        f"def f(x: NDArray[Float, {size}]):\n"
        f"    y = np.{elem_op}(x * x + 1.0)\n"
        f"    out = np.{red_op}(y)\n"
        f"    diff = out - ({ref!r})\n"
        f"    assert diff < {FLOAT_TOL!r}\n"
        f"    assert diff > {-FLOAT_TOL!r}\n"
    )
    return source, {"x": x.tolist()}, "B", ref, True


def _gen_shape_c(rng: random.Random) -> tuple[str, dict, str, object, bool]:
    """Shape C: matmul on float matrices."""
    M = rng.choice(MATMUL_SIZES)
    K = rng.choice(MATMUL_SIZES)
    N = rng.choice(MATMUL_SIZES)
    a = _numpy.array(
        [[rng.uniform(FLOAT_LO, FLOAT_HI) for _ in range(K)] for _ in range(M)],
        dtype=_numpy.float64,
    )
    b = _numpy.array(
        [[rng.uniform(FLOAT_LO, FLOAT_HI) for _ in range(N)] for _ in range(K)],
        dtype=_numpy.float64,
    )
    c_ref = _numpy.matmul(a, b)
    assert_lines = []
    for i in range(M):
        for j in range(N):
            ref_ij = float(c_ref[i, j])
            assert_lines.append(f"    d = c[{i}, {j}] - ({ref_ij!r})")
            assert_lines.append(f"    assert d < {FLOAT_TOL!r}")
            assert_lines.append(f"    assert d > {-FLOAT_TOL!r}")
    assertions = "\n".join(assert_lines)
    source = (
        "from zinnia import zk_circuit\n"
        "from zinnia.lang.operator import NDArray, Integer, Float\n"
        "import numpy as np\n"
        "\n"
        "@zk_circuit\n"
        f"def f(a: NDArray[Float, {M}, {K}], b: NDArray[Float, {K}, {N}]):\n"
        f"    c = np.matmul(a, b)\n"
        f"{assertions}\n"
    )
    return source, {"a": a.tolist(), "b": b.tolist()}, "C", c_ref.tolist(), True


_PROLOGUE = (
    "from zinnia import zk_circuit, requires\n"
    "from zinnia.lang.operator import NDArray, Integer, Float\n"
    "import numpy as np\n"
    "\n"
)


def _float_assert_lines(expr: str, ref: float, tol: float = FLOAT_TOL) -> str:
    """Helper: emit `assert |expr - ref| < tol` as two-sided inequality."""
    return (
        f"    _d = {expr} - ({ref!r})\n"
        f"    assert _d < {tol!r}\n"
        f"    assert _d > {-tol!r}\n"
    )


def _gen_shape_d(rng: random.Random) -> tuple[str, dict, str, object, bool]:
    """Shape D: indexing."""
    flavor = rng.choice(["scalar", "slice", "fancy", "take"])
    is_float = rng.random() < 0.5
    size = rng.choice([6, 8, 12])
    if is_float:
        x = _numpy.array(
            [rng.uniform(FLOAT_LO, FLOAT_HI) for _ in range(size)],
            dtype=_numpy.float64,
        )
        ann = f"NDArray[Float, {size}]"
    else:
        x = _numpy.array(
            [rng.randint(INT_LO, INT_HI) for _ in range(size)],
            dtype=_numpy.int64,
        )
        ann = f"NDArray[Integer, {size}]"

    if flavor == "scalar":
        i = rng.randint(0, size - 1)
        body = f"    out = x[{i}]\n"
        ref_val = x[i].item()
        if is_float:
            asserts = _float_assert_lines("out", float(ref_val))
        else:
            asserts = f"    assert out == {int(ref_val)}\n"
    elif flavor == "slice":
        start = rng.randint(0, size - 2)
        stop = rng.randint(start + 1, size)
        red_op = rng.choice(REDUCTIONS_FLOAT if is_float else REDUCTIONS_INT)
        body = (
            f"    sub = x[{start}:{stop}]\n"
            f"    out = np.{red_op}(sub)\n"
        )
        ref = getattr(_numpy, red_op)(x[start:stop])
        if is_float or red_op == "mean":
            asserts = _float_assert_lines("out", float(ref))
            ref_val = float(ref)
        else:
            ref_val = int(ref)
            asserts = f"    assert out == {ref_val}\n"
    elif flavor == "fancy":
        k = rng.randint(2, min(4, size))
        idx = [rng.randint(0, size - 1) for _ in range(k)]
        red_op = rng.choice(REDUCTIONS_FLOAT if is_float else REDUCTIONS_INT)
        body = (
            f"    idx = np.asarray({idx})\n"
            f"    sub = x[idx]\n"
            f"    out = np.{red_op}(sub)\n"
        )
        ref = getattr(_numpy, red_op)(x[_numpy.asarray(idx)])
        if is_float or red_op == "mean":
            asserts = _float_assert_lines("out", float(ref))
            ref_val = float(ref)
        else:
            ref_val = int(ref)
            asserts = f"    assert out == {ref_val}\n"
    else:  # take
        k = rng.randint(2, min(4, size))
        idx = [rng.randint(0, size - 1) for _ in range(k)]
        red_op = rng.choice(REDUCTIONS_FLOAT if is_float else REDUCTIONS_INT)
        body = (
            f"    idx = np.asarray({idx})\n"
            f"    sub = np.take(x, idx)\n"
            f"    out = np.{red_op}(sub)\n"
        )
        ref = getattr(_numpy, red_op)(_numpy.take(x, _numpy.asarray(idx)))
        if is_float or red_op == "mean":
            asserts = _float_assert_lines("out", float(ref))
            ref_val = float(ref)
        else:
            ref_val = int(ref)
            asserts = f"    assert out == {ref_val}\n"

    source = (
        _PROLOGUE
        + "@zk_circuit\n"
        + f"def f(x: {ann}):\n"
        + body
        + asserts
    )
    return source, {"x": x.tolist()}, "D", ref_val, True


def _gen_shape_e(rng: random.Random) -> tuple[str, dict, str, object, bool]:
    """Shape E: shape-preserving op chains on a 2D array → reduce."""
    M = rng.choice([2, 3, 4])
    N = rng.choice([2, 3, 4])
    is_float = rng.random() < 0.5
    if is_float:
        a = _numpy.array(
            [[rng.uniform(FLOAT_LO, FLOAT_HI) for _ in range(N)] for _ in range(M)],
            dtype=_numpy.float64,
        )
        ann = f"NDArray[Float, {M}, {N}]"
    else:
        a = _numpy.array(
            [[rng.randint(INT_LO, INT_HI) for _ in range(N)] for _ in range(M)],
            dtype=_numpy.int64,
        )
        ann = f"NDArray[Integer, {M}, {N}]"

    # Choose one op that's safe given (M,N).
    op_choices = ["transpose", "swapaxes", "flatten", "reshape"]
    if M == N:
        # Reshape squared works fine; nothing exclusive here.
        pass
    op = rng.choice(op_choices)
    if op == "transpose":
        body = "    y = np.transpose(a)\n"
        y = _numpy.transpose(a)
    elif op == "swapaxes":
        body = "    y = np.swapaxes(a, 0, 1)\n"
        y = _numpy.swapaxes(a, 0, 1)
    elif op == "flatten":
        body = "    y = a.flatten()\n"
        y = a.flatten()
    else:  # reshape — flatten to 1-D
        body = f"    y = np.reshape(a, ({M * N},))\n"
        y = _numpy.reshape(a, (M * N,))

    # Optional second shape-preserving op chained on top.
    if rng.random() < 0.5:
        if y.ndim == 1:
            body += "    y = np.expand_dims(y, 0)\n"
            y = _numpy.expand_dims(y, 0)
        else:
            body += "    y = np.transpose(y)\n"
            y = _numpy.transpose(y)

    red_op = rng.choice(REDUCTIONS_FLOAT if is_float else REDUCTIONS_INT)
    body += f"    out = np.{red_op}(y)\n"
    ref = getattr(_numpy, red_op)(y)
    if is_float or red_op == "mean":
        ref_val = float(ref)
        asserts = _float_assert_lines("out", ref_val)
    else:
        ref_val = int(ref)
        asserts = f"    assert out == {ref_val}\n"
    source = (
        _PROLOGUE
        + "@zk_circuit\n"
        + f"def f(a: {ann}):\n"
        + body
        + asserts
    )
    return source, {"a": a.tolist()}, "E", ref_val, True


def _gen_shape_f(rng: random.Random) -> tuple[str, dict, str, object, bool]:
    """Shape F: np.where(cond, a, b)."""
    size = rng.choice(SHAPE_SIZES)
    is_float = rng.random() < 0.5
    if is_float:
        a = _numpy.array(
            [rng.uniform(FLOAT_LO, FLOAT_HI) for _ in range(size)],
            dtype=_numpy.float64,
        )
        b = _numpy.array(
            [rng.uniform(FLOAT_LO, FLOAT_HI) for _ in range(size)],
            dtype=_numpy.float64,
        )
        ann_a = f"NDArray[Float, {size}]"
        ann_b = f"NDArray[Float, {size}]"
    else:
        a = _numpy.array(
            [rng.randint(INT_LO, INT_HI) for _ in range(size)],
            dtype=_numpy.int64,
        )
        b = _numpy.array(
            [rng.randint(INT_LO, INT_HI) for _ in range(size)],
            dtype=_numpy.int64,
        )
        ann_a = f"NDArray[Integer, {size}]"
        ann_b = f"NDArray[Integer, {size}]"

    # Cond either a literal mask or computed from `a > 0`.
    if rng.random() < 0.5:
        cond_src = "a > 0"
        cond = a > 0
    else:
        mask = [rng.choice([True, False]) for _ in range(size)]
        cond_src = f"np.asarray({mask})"
        cond = _numpy.asarray(mask)

    red_op = rng.choice(REDUCTIONS_FLOAT if is_float else REDUCTIONS_INT)
    y = _numpy.where(cond, a, b)
    ref = getattr(_numpy, red_op)(y)
    body = (
        f"    cond = {cond_src}\n"
        f"    y = np.where(cond, a, b)\n"
        f"    out = np.{red_op}(y)\n"
    )
    if is_float or red_op == "mean":
        ref_val = float(ref)
        asserts = _float_assert_lines("out", ref_val)
    else:
        ref_val = int(ref)
        asserts = f"    assert out == {ref_val}\n"
    source = (
        _PROLOGUE
        + "@zk_circuit\n"
        + f"def f(a: {ann_a}, b: {ann_b}):\n"
        + body
        + asserts
    )
    return source, {"a": a.tolist(), "b": b.tolist()}, "F", ref_val, True


def _gen_shape_g(rng: random.Random) -> tuple[str, dict, str, object, bool]:
    """Shape G: concat/stack on two 1-D arrays → reduce."""
    size = rng.choice([2, 3, 4])
    is_float = rng.random() < 0.5
    if is_float:
        a = _numpy.array(
            [rng.uniform(FLOAT_LO, FLOAT_HI) for _ in range(size)],
            dtype=_numpy.float64,
        )
        b = _numpy.array(
            [rng.uniform(FLOAT_LO, FLOAT_HI) for _ in range(size)],
            dtype=_numpy.float64,
        )
        ann = f"NDArray[Float, {size}]"
    else:
        a = _numpy.array(
            [rng.randint(INT_LO, INT_HI) for _ in range(size)],
            dtype=_numpy.int64,
        )
        b = _numpy.array(
            [rng.randint(INT_LO, INT_HI) for _ in range(size)],
            dtype=_numpy.int64,
        )
        ann = f"NDArray[Integer, {size}]"

    op = rng.choice(["concatenate", "stack", "vstack", "hstack"])
    if op == "concatenate":
        body = "    y = np.concatenate([a, b])\n"
        y = _numpy.concatenate([a, b])
    elif op == "stack":
        body = "    y = np.stack([a, b])\n"
        y = _numpy.stack([a, b])
    elif op == "vstack":
        body = "    y = np.vstack([a, b])\n"
        y = _numpy.vstack([a, b])
    else:  # hstack
        body = "    y = np.hstack([a, b])\n"
        y = _numpy.hstack([a, b])

    red_op = rng.choice(REDUCTIONS_FLOAT if is_float else REDUCTIONS_INT)
    body += f"    out = np.{red_op}(y)\n"
    ref = getattr(_numpy, red_op)(y)
    if is_float or red_op == "mean":
        ref_val = float(ref)
        asserts = _float_assert_lines("out", ref_val)
    else:
        ref_val = int(ref)
        asserts = f"    assert out == {ref_val}\n"

    source = (
        _PROLOGUE
        + "@zk_circuit\n"
        + f"def f(a: {ann}, b: {ann}):\n"
        + body
        + asserts
    )
    return source, {"a": a.tolist(), "b": b.tolist()}, "G", ref_val, True


def _gen_shape_h(rng: random.Random) -> tuple[str, dict, str, object, bool]:
    """Shape H: deeper compositions mixing 3-5 ops from D/E/F/G."""
    # Two parallel arrays for compositional flexibility.
    M = rng.choice([2, 3, 4])
    N = rng.choice([2, 3, 4])
    is_float = rng.random() < 0.5
    if is_float:
        a = _numpy.array(
            [[rng.uniform(FLOAT_LO, FLOAT_HI) for _ in range(N)] for _ in range(M)],
            dtype=_numpy.float64,
        )
        b = _numpy.array(
            [[rng.uniform(FLOAT_LO, FLOAT_HI) for _ in range(M)] for _ in range(N)],
            dtype=_numpy.float64,
        )
        ann_a = f"NDArray[Float, {M}, {N}]"
        ann_b = f"NDArray[Float, {N}, {M}]"
    else:
        a = _numpy.array(
            [[rng.randint(INT_LO, INT_HI) for _ in range(N)] for _ in range(M)],
            dtype=_numpy.int64,
        )
        b = _numpy.array(
            [[rng.randint(INT_LO, INT_HI) for _ in range(M)] for _ in range(N)],
            dtype=_numpy.int64,
        )
        ann_a = f"NDArray[Integer, {M}, {N}]"
        ann_b = f"NDArray[Integer, {N}, {M}]"

    # Build a chain of 3-5 ops on a (using b inside matmul when chosen).
    n_ops = rng.randint(3, 5)
    body_lines: list[str] = ["    y = a\n"]
    y = a
    op_pool = ["transpose", "reshape_flatten", "swapaxes", "expand_dims", "squeeze",
               "matmul_b", "where_self", "slice", "concat_self"]

    for _ in range(n_ops):
        # Filter ops by current shape compatibility.
        choices = []
        if y.ndim == 2:
            choices += ["transpose", "swapaxes", "reshape_flatten", "matmul_b",
                        "expand_dims", "where_self"]
        elif y.ndim == 1:
            choices += ["expand_dims", "slice", "concat_self", "where_self"]
        elif y.ndim == 3:
            choices += ["squeeze", "reshape_flatten"]
        if not choices:
            break
        op = rng.choice(choices)
        if op == "transpose":
            body_lines.append("    y = np.transpose(y)\n")
            y = _numpy.transpose(y)
        elif op == "swapaxes":
            body_lines.append("    y = np.swapaxes(y, 0, 1)\n")
            y = _numpy.swapaxes(y, 0, 1)
        elif op == "reshape_flatten":
            body_lines.append(f"    y = np.reshape(y, ({y.size},))\n")
            y = _numpy.reshape(y, (y.size,))
        elif op == "matmul_b":
            # Only valid if y has 2 dims and y.shape[-1] == b.shape[0].
            if y.ndim == 2 and y.shape[-1] == b.shape[0]:
                body_lines.append("    y = np.matmul(y, b)\n")
                y = _numpy.matmul(y, b)
            else:
                # Skip — pick a no-op alt.
                body_lines.append("    y = np.transpose(y)\n")
                y = _numpy.transpose(y)
        elif op == "expand_dims":
            body_lines.append("    y = np.expand_dims(y, 0)\n")
            y = _numpy.expand_dims(y, 0)
        elif op == "squeeze":
            if 1 in y.shape:
                body_lines.append("    y = np.squeeze(y)\n")
                y = _numpy.squeeze(y)
            else:
                body_lines.append("    y = np.transpose(y)\n")
                y = _numpy.transpose(y)
        elif op == "where_self":
            # Replace negatives with zero.
            body_lines.append("    y = np.where(y > 0, y, y * 0)\n")
            y = _numpy.where(y > 0, y, y * 0)
        elif op == "slice":
            if y.ndim == 1 and y.shape[0] >= 2:
                stop = max(2, y.shape[0] // 2 + 1)
                body_lines.append(f"    y = y[0:{stop}]\n")
                y = y[0:stop]
        elif op == "concat_self":
            if y.ndim == 1:
                body_lines.append("    y = np.concatenate([y, y])\n")
                y = _numpy.concatenate([y, y])

    red_op = rng.choice(REDUCTIONS_FLOAT if is_float else REDUCTIONS_INT)
    body_lines.append(f"    out = np.{red_op}(y)\n")
    ref = getattr(_numpy, red_op)(y)
    if is_float or red_op == "mean":
        ref_val = float(ref)
        asserts = _float_assert_lines("out", ref_val, tol=1e-2)
    else:
        ref_val = int(ref)
        asserts = f"    assert out == {ref_val}\n"

    source = (
        _PROLOGUE
        + "@zk_circuit\n"
        + f"def f(a: {ann_a}, b: {ann_b}):\n"
        + "".join(body_lines)
        + asserts
    )
    return source, {"a": a.tolist(), "b": b.tolist()}, "H", ref_val, True


def _gen_shape_i(rng: random.Random) -> tuple[str, dict, str, object, bool]:
    """Shape I: programs with @requires on a scalar input.

    The body uses an op that benefits from a precondition (sqrt for x >= 0,
    log for x >= 1). Inputs are sampled from a wider range than the
    precondition allows; whether they satisfy is recorded in
    `expected_satisfied`.
    """
    variant = rng.choice(["sqrt_nonneg", "log_pos", "range_clamp"])

    if variant == "sqrt_nonneg":
        # Precondition: x >= 0. Input drawn from [-3, 10].
        x_val = rng.randint(-3, 10)
        cond_src = "lambda x: x >= 0"
        satisfies = x_val >= 0
        # ref is np.sqrt(x) when x >= 0; for violating we still set the assert
        # against the satisfying-value formula so the precondition-fail path is
        # the only thing that triggers unsatisfied.
        ref_val = float(_numpy.sqrt(max(0, x_val)))
        body = "    y = np.sqrt(x)\n"
        ann = "x: int"
    elif variant == "log_pos":
        # Precondition: x >= 1. Input drawn from [-2, 10].
        x_val = rng.randint(-2, 10)
        cond_src = "lambda x: x >= 1"
        satisfies = x_val >= 1
        ref_val = float(_numpy.log(max(1, x_val)))
        body = "    y = np.log(x)\n"
        ann = "x: int"
    else:  # range_clamp — a precondition without an op-domain hook.
        x_val = rng.randint(-20, 20)
        lo = rng.randint(0, 5)
        hi = rng.randint(lo + 1, 10)
        cond_src = f"lambda x: {lo} <= x <= {hi}"
        satisfies = lo <= x_val <= hi
        # Body: just emit an assert on a derived value.
        ref_val = float(x_val * 2)
        body = "    y = x * 2\n"
        ann = "x: int"

    # Assert that y is close to ref_val. When satisfies=True, ref_val is the
    # numpy reference, so the assert holds. When satisfies=False, the prover
    # should refuse via the @requires witness check; the assert content is
    # irrelevant (it will never be evaluated for a valid witness).
    if variant == "range_clamp":
        # y is an int.
        asserts = f"    assert y == {int(ref_val)}\n"
    else:
        asserts = _float_assert_lines("y", ref_val)

    source = (
        _PROLOGUE
        + "@zk_circuit\n"
        + f"@requires({cond_src})\n"
        + f"def f({ann}):\n"
        + body
        + asserts
    )
    return source, {"x": x_val}, "I", ref_val, satisfies


SHAPE_GENERATORS = {
    "A": _gen_shape_a,
    "B": _gen_shape_b,
    "C": _gen_shape_c,
    "D": _gen_shape_d,
    "E": _gen_shape_e,
    "F": _gen_shape_f,
    "G": _gen_shape_g,
    "H": _gen_shape_h,
    "I": _gen_shape_i,
}


def gen_program(shape: str, rng: random.Random):
    return SHAPE_GENERATORS[shape](rng)


# ----------------------------------------------------------------------------
# Runner — child process so a panic / hang doesn't kill the orchestrator
# ----------------------------------------------------------------------------

def _zinnia_child_worker(source: str, inputs_repr: dict, sys_path: list, sys_cwd: str, conn) -> None:
    """Run inside a `multiprocessing` child. Compiles + executes the
    circuit; reports back via ``conn``.

    The pipe payload is one of:
      ("ok", {"satisfied": bool})
      ("fail", {"kind": "compile_failure", "exception": str})
    """
    try:
        # spawn-method children start with a fresh sys.path; replay the
        # parent's so `import zinnia` (and any sibling packages on the
        # path) keeps working. Also inject the parent's cwd so that bare
        # `import zinnia` from the zinnia-src checkout resolves.
        import sys as _sys
        import os as _os
        try:
            _os.chdir(sys_cwd)
        except Exception:
            pass
        for p in reversed(sys_path):
            if p == "":
                p = sys_cwd
            if p and p not in _sys.path:
                _sys.path.insert(0, p)

        # Register the source with linecache so `inspect.getsource` (used by
        # @zk_circuit at decoration time) can recover the function body even
        # though it was injected via ``exec``.
        fake_path = "<fuzz_source>"
        linecache.cache[fake_path] = (
            len(source),
            None,
            source.splitlines(keepends=True),
            fake_path,
        )
        code = compile(source, fake_path, "exec")

        import numpy as np  # type: ignore
        from zinnia import zk_circuit  # type: ignore
        from zinnia.lang.operator import NDArray, Integer, Float  # type: ignore

        g: dict = {
            "__name__": "__fuzz__",
            "np": np,
            "zk_circuit": zk_circuit,
            "NDArray": NDArray,
            "Integer": Integer,
            "Float": Float,
        }
        exec(code, g)  # populates g["f"]
        f = g["f"]
        # Convert each input — arrays via np.asarray, scalars (int/float) kept
        # as-is. Pickle round-trip during pipe send preserves the python type.
        np_inputs: dict = {}
        for k, v in inputs_repr.items():
            if isinstance(v, list):
                np_inputs[k] = np.asarray(v)
            else:
                np_inputs[k] = v
        # Generators use one of: {x}, {a}, {a, b}. Dispatch by key set.
        keys = set(np_inputs)
        if keys == {"x"}:
            result = f(np_inputs["x"])
        elif keys == {"a"}:
            result = f(np_inputs["a"])
        elif keys == {"a", "b"}:
            result = f(np_inputs["a"], np_inputs["b"])
        else:
            raise RuntimeError(f"unknown input shape: {list(np_inputs)}")
        satisfied = bool(getattr(result, "satisfied", False))
        conn.send(("ok", {"satisfied": satisfied}))
    except BaseException as e:  # noqa: BLE001 — PyO3 panics are BaseException
        tb = traceback.format_exc()
        conn.send((
            "fail",
            {
                "kind": "compile_failure",
                "exception": f"{type(e).__name__}: {e}",
                "traceback": tb[-2000:],
            },
        ))
    finally:
        try:
            conn.close()
        except Exception:
            pass


def run_zinnia(source: str, inputs_repr: dict, timeout: float = TIMEOUT_SEC) -> dict:
    """Run the Zinnia circuit in a watchdog-protected child process.

    Returns a dict with at least:
      {"status": "ok", "satisfied": bool}
      {"status": "compile_failure", "exception": str, "traceback": str}
      {"status": "timeout"}
      {"status": "crash", "exception": str}
    """
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_zinnia_child_worker,
        args=(source, inputs_repr, list(sys.path), os.getcwd(), child_conn),
    )
    proc.start()
    child_conn.close()  # parent only reads
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            proc.kill()
            proc.join()
        return {"status": "timeout"}
    # Drain the pipe.
    if parent_conn.poll():
        try:
            tag, payload = parent_conn.recv()
        except EOFError:
            return {"status": "crash", "exception": f"child died (exitcode={proc.exitcode})"}
        if tag == "ok":
            return {"status": "ok", "satisfied": payload["satisfied"]}
        elif tag == "fail":
            return {"status": "compile_failure", **payload}
        else:
            return {"status": "crash", "exception": f"unknown tag: {tag!r}"}
    return {"status": "crash", "exception": f"no message; exitcode={proc.exitcode}"}


# ----------------------------------------------------------------------------
# Comparator
# ----------------------------------------------------------------------------

def interpret_outcome(
    zinnia_result: dict,
    expected_satisfied: bool = True,
) -> tuple[str, str | None]:
    """Map a Zinnia child outcome onto one of:
      ("success", None) — observed satisfied matches expected
      ("divergence", <detail>) — observed satisfied differs from expected
      ("compile_failure", <exception_str>)
      ("timeout", None)
      ("crash", <exception_str>)
    """
    status = zinnia_result["status"]
    if status == "ok":
        observed = bool(zinnia_result["satisfied"])
        if observed == expected_satisfied:
            return ("success", None)
        if expected_satisfied:
            return ("divergence", "assertion-unsatisfied")
        # Expected refusal but witness accepted — Phase E discharge failure.
        return ("divergence", "witness-check-missed")
    if status == "compile_failure":
        return ("compile_failure", zinnia_result.get("exception"))
    if status == "timeout":
        return ("timeout", None)
    return ("crash", zinnia_result.get("exception"))


# ----------------------------------------------------------------------------
# Reporter
# ----------------------------------------------------------------------------

def write_report(
    report_dir: Path,
    index: int,
    shape: str,
    source: str,
    inputs: dict,
    ref_output,
    zinnia_result: dict,
    outcome_kind: str,
    detail: str | None,
) -> Path:
    payload = {
        "index": index,
        "shape": shape,
        "kind": outcome_kind,
        "detail": detail,
        "source": source,
        "inputs": inputs,
        "ref_output": ref_output,
        "zinnia": {
            "status": zinnia_result.get("status"),
            "satisfied": zinnia_result.get("satisfied"),
            "exception": zinnia_result.get("exception"),
        },
    }
    if "traceback" in zinnia_result:
        payload["zinnia"]["traceback"] = zinnia_result["traceback"]
    out_path = report_dir / f"{index:04d}-{outcome_kind}-{shape}.json"
    out_path.write_text(json.dumps(payload, indent=2, default=_json_default))
    return out_path


def _json_default(o):
    if isinstance(o, _numpy.ndarray):
        return o.tolist()
    if isinstance(o, (_numpy.integer,)):
        return int(o)
    if isinstance(o, (_numpy.floating,)):
        return float(o)
    return str(o)


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--shape",
        choices=list(SHAPE_GENERATORS.keys()),
        default=None,
        help="restrict generator to a single shape",
    )
    parser.add_argument("--timeout", type=float, default=TIMEOUT_SEC)
    args = parser.parse_args(argv)

    seed = args.seed if args.seed is not None else int(time.time())
    rng = random.Random(seed)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    report_dir = REPORT_ROOT / timestamp
    report_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "success": 0,
        "divergence": 0,
        "compile_failure": 0,
        "timeout": 0,
        "crash": 0,
    }
    per_shape = {k: dict(summary) for k in SHAPE_GENERATORS.keys()}
    # Weight the new shapes a little lower than the original A/B/C so the
    # report retains a decent baseline of the existing surface; I (requires)
    # is the most expensive (Phase E) so it gets the smallest weight.
    shape_weights = {
        "A": 3, "B": 3, "C": 3,
        "D": 3, "E": 3, "F": 3, "G": 3, "H": 3,
        "I": 2,
    }
    shape_population = list(shape_weights.keys())
    shape_weight_list = [shape_weights[s] for s in shape_population]

    print(f"[fuzz] seed={seed} iterations={args.iterations} report_dir={report_dir}")
    start = time.time()

    for i in range(args.iterations):
        if args.shape:
            shape = args.shape
        else:
            shape = rng.choices(shape_population, weights=shape_weight_list, k=1)[0]
        try:
            source, inputs, shape_tag, ref_output, expected_satisfied = gen_program(shape, rng)
        except Exception as e:
            print(f"[fuzz] iter={i} GEN-ERROR shape={shape} {type(e).__name__}: {e}")
            continue

        zinnia_result = run_zinnia(source, inputs, timeout=args.timeout)
        outcome_kind, detail = interpret_outcome(zinnia_result, expected_satisfied)

        summary[outcome_kind] += 1
        per_shape[shape_tag][outcome_kind] += 1

        print(
            f"[fuzz] iter={i:04d} shape={shape_tag} "
            f"outcome={outcome_kind}"
            + (f" expected_sat={expected_satisfied}" if shape_tag == "I" else "")
            + (f" detail={detail}" if detail else "")
        )

        if outcome_kind != "success":
            write_report(
                report_dir=report_dir,
                index=i,
                shape=shape_tag,
                source=source,
                inputs=inputs,
                ref_output=ref_output,
                zinnia_result=zinnia_result,
                outcome_kind=outcome_kind,
                detail=detail,
            )

    elapsed = time.time() - start

    print()
    print(f"[fuzz] done in {elapsed:.1f}s")
    print(f"[fuzz] summary: {summary}")
    for shp, sub in per_shape.items():
        print(f"[fuzz]   shape {shp}: {sub}")
    print(f"[fuzz] reports: {report_dir}")

    # Always write a summary.json so the report dir has something even on
    # an all-success run.
    (report_dir / "summary.json").write_text(json.dumps({
        "seed": seed,
        "iterations": args.iterations,
        "elapsed_sec": elapsed,
        "summary": summary,
        "per_shape": per_shape,
    }, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
