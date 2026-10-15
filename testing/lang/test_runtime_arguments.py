"""Arguments known only at run time: used as they are, or rejected.

np.linspace read a `start` or `stop` known only at run time as 0.0, and an
integer dtype truncated the samples where NumPy 2 floors them. Elsewhere an
option known only at run time (`endpoint=`, `num=` with array endpoints,
`axis=`, `k=`, `M=`, `keepdims=`, the `start` of `list.index`) was replaced by
its default, and `np.linspace(..., retstep=True)`, a positional `endpoint` and
`np.concatenate(..., axis=None)` were ignored. Each is now computed as NumPy
(or Python) computes it or rejected at compile time: an option that decides
an output's shape must be known at compile time, while the `start` and `stop`
of `list.index` only decide which elements are searched and are compared at
run time.

Every accepted kernel is checked against NumPy on the mock, and halo2-base and
Noir must prove the mock's values.
"""
import numpy as np
import pytest

from testing.lang.all_backends import assert_all_backends_match_numpy
from testing.lang.mock_outputs import compile_kernel

# np.linspace with runtime endpoints, integer dtypes and a runtime num.
RUNTIME_ENDPOINTS = {
    "runtime start": ("def kernel(c: Float):\n    return np.linspace(c, 4.0, 3)\n", (1.0,)),
    "runtime stop": ("def kernel(c: Float):\n    return np.linspace(0.0, c, 5)\n", (3.0,)),
    "runtime start and stop": ("def kernel(c: Float, d: Float):\n    return np.linspace(c, d, 4)\n", (2.5, -1.0)),
    "integer runtime endpoints": ("def kernel(c: Integer, d: Integer):\n    return np.linspace(c, d, 5)\n", (-3, 7)),
    "runtime start, no endpoint": ("def kernel(c: Float):\n    return np.linspace(c, 1.0, 4, endpoint=False)\n", (-1.0,)),
    "runtime start, positional endpoint": ("def kernel(c: Float):\n    return np.linspace(c, 1.0, 4, False)\n", (-1.0,)),
    "runtime start, int dtype": ("def kernel(c: Float):\n    return np.linspace(c, 1.0, 4, dtype=np.int64)\n", (-1.0,)),
    "runtime start and num": (
        "@requires(lambda n, c: 2 <= n <= 6)\ndef kernel(n: Integer, c: Float):\n"
        "    x = np.linspace(c, c + 1.0, n)\n    return x[0], x[n - 1], np.max(x), np.sum(x)\n",
        (4, 1.5),
    ),
    "runtime stop and num, int dtype": (
        "@requires(lambda n, c: 2 <= n <= 6)\ndef kernel(n: Integer, c: Integer):\n"
        "    x = np.linspace(-2, c, n, dtype=np.int64)\n    return x[0], x[1], x[n - 1], np.sum(x)\n",
        (5, 1),
    ),
    "constant endpoints, int dtype, negative samples": ("def kernel():\n    return np.linspace(-1, 0, 3, dtype=np.int64)\n", ()),
    "constant endpoints, int dtype, inexact last sample": ("def kernel():\n    return np.linspace(0, 1, 4, dtype=np.int64)\n", ()),
    "runtime num, int dtype, inexact last sample": (
        "@requires(lambda n: 2 <= n <= 6)\ndef kernel(n: Integer):\n"
        "    x = np.linspace(0, 1, n, dtype=np.int64)\n    return x[n - 1], np.sum(x)\n",
        (4,),
    ),
    "array endpoints, int dtype": (
        "def kernel(c: Float):\n    return np.linspace(np.array([-1.0, 0.0]), np.array([0.0, 1.0]), 3, dtype=np.int64)\n",
        (0.0,),
    ),
}


@pytest.mark.parametrize("source,args", RUNTIME_ENDPOINTS.values(), ids=RUNTIME_ENDPOINTS.keys())
def test_linspace_matches_numpy(source, args):
    assert_all_backends_match_numpy(source, *args)


# Arguments that must be known at compile time are rejected, not replaced by
# their default.
REJECTED = {
    "linspace endpoint= at run time": (
        "def kernel(e: Boolean):\n    return np.linspace(0.0, 1.0, 4, endpoint=e)\n",
        "endpoint",
    ),
    "linspace with array endpoints, num at run time": (
        "@requires(lambda n: 2 <= n <= 6)\ndef kernel(n: Integer):\n"
        "    return np.linspace(np.array([0.0, 1.0]), np.array([1.0, 2.0]), n)\n",
        "num",
    ),
    "linspace retstep=True": ("def kernel():\n    return np.linspace(0.0, 1.0, 5, retstep=True)\n", "retstep"),
    "linspace unknown keyword": ("def kernel():\n    return np.linspace(0.0, 1.0, 5, foo=3)\n", "foo"),
    "diag k= at run time": ("def kernel(a: Integer[Array, '3'], k: Integer):\n    return np.diag(a, k)\n", "k"),
    "DynamicNDArray.eye M at run time": (
        "def kernel(m: Integer):\n    return np.sum(DynamicNDArray.eye(3, m))\n",
        "M",
    ),
    "concatenate axis= at run time": (
        "def kernel(m: Integer[Array, '2 2'], a: Integer):\n    return np.concatenate([m, m], axis=a)\n",
        "axis",
    ),
    "rot90 k at run time": ("def kernel(m: Integer[Array, '2 2'], k: Integer):\n    return np.rot90(m, k)\n", "k"),
    "keepdims= at run time": (
        "def kernel(m: Integer[Array, '2 2'], k: Boolean):\n    return np.sum(m, axis=0, keepdims=k)\n",
        "keepdims",
    ),
}


@pytest.mark.parametrize("source,needle", REJECTED.values(), ids=REJECTED.keys())
def test_runtime_option_is_rejected(source, needle):
    with pytest.raises(Exception, match=needle):
        compile_kernel(source)


# Compile-time options that used to be ignored.
OPTIONS = {
    "concatenate axis=None": (
        "def kernel(m: Integer[Array, '2 2']):\n    return np.concatenate([m, m], axis=None)\n",
        (np.array([[1, 2], [3, 4]]),),
    ),
    "concatenate positional None": (
        "def kernel(m: Integer[Array, '2 2']):\n    return np.concatenate((m, m), None)\n",
        (np.array([[1, 2], [3, 4]]),),
    ),
    "rot90 twice": ("def kernel(m: Integer[Array, '2 3']):\n    return np.rot90(m, 2)\n", (np.arange(6).reshape(2, 3),)),
    "rot90 axes (1, 0)": (
        "def kernel(m: Integer[Array, '2 3']):\n    return np.rot90(m, 1, (1, 0))\n",
        (np.arange(6).reshape(2, 3),),
    ),
    "list.index with start and stop": ("def kernel(x: Integer):\n    l = [x, 2, x, 2, x]\n    return l.index(x, 1, 4)\n", (7,)),
    "list.index with a negative start": ("def kernel(x: Integer):\n    l = [x, 2, x, 2]\n    return l.index(2, -1)\n", (7,)),
}


@pytest.mark.parametrize("source,args", OPTIONS.values(), ids=OPTIONS.keys())
def test_compile_time_option_matches_numpy(source, args):
    assert_all_backends_match_numpy(source, *args)


def test_list_index_stop_excludes_a_later_match():
    """`[1, 2, 3].index(3, 0, 2)` raises in Python: no proof."""
    program = compile_kernel("def kernel(x: Integer):\n    l = [1, 2, x]\n    return l.index(x, 0, 2)\n")
    assert program.prove(3, backend="mock").proof_bytes_hex != "mock_satisfied"


# `list.index` with `start` and `stop` known only at run time searches the
# window Python's slice bounds select (negative from the end, clamped).
RUNTIME_LIST_INDEX_BOUNDS = {
    "start": ("def kernel(s: Integer):\n    l = [1, 2, 1, 2]\n    return l.index(1, s)\n", (1,)),
    "negative start": ("def kernel(s: Integer):\n    l = [1, 2, 1, 2]\n    return l.index(1, s)\n", (-2,)),
    "start before the list": ("def kernel(s: Integer):\n    l = [1, 2, 1, 2]\n    return l.index(2, s)\n", (-9,)),
    "start and stop": (
        "def kernel(x: Integer, s: Integer, e: Integer):\n    l = [x, 2, x, 2, x]\n    return l.index(x, s, e)\n",
        (7, 1, 4),
    ),
    "stop past the end": (
        "def kernel(s: Integer, e: Integer):\n    l = [1, 2, 1, 2]\n    return l.index(2, s, e)\n",
        (2, 99),
    ),
}


@pytest.mark.parametrize("source,args", RUNTIME_LIST_INDEX_BOUNDS.values(), ids=RUNTIME_LIST_INDEX_BOUNDS.keys())
def test_list_index_runtime_bounds_match_python(source, args):
    assert_all_backends_match_numpy(source, *args)


@pytest.mark.parametrize("start,stop", [(3, 4), (0, 0), (-1, 4), (2, 2), (1, -2)])
def test_list_index_runtime_bounds_exclude_matches_outside(start, stop):
    """`[1, 2, 1, 2].index(1, start, stop)` raises in Python when no 1 lies in
    `[start:stop]`: no proof."""
    program = compile_kernel(
        "def kernel(s: Integer, e: Integer):\n    l = [1, 2, 1, 2]\n    return l.index(1, s, e)\n"
    )
    assert program.prove(start, stop, backend="mock").proof_bytes_hex != "mock_satisfied"
