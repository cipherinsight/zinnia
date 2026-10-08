"""Facts about an array's elements lapse when the array is written.

`np.zeros`, `np.ones`, `np.full`, `np.identity` and `np.eye` state a fact about
every element of their result (`forall_eq_const`, `is_identity`), and shape
operations (slices, `reshape`, `concatenate`, `tile`, ...) relay it to their
result. Reductions, `where` and `matmul` then skip the computation
(`np.sum(np.zeros(n))` is the constant 0). The fact is keyed by the array's
value id, which a write through another name for the same array
(`y = x; y[0] = 5`) keeps, so the shortcut used to read elements the array no
longer held. The compiler now records what the array held when the fact was
stated and leaves the fact out once the array is written
(`src/builder/array_facts.rs`).

Every kernel is checked against NumPy on the mock, and halo2-base and Noir must
prove the mock's values.
"""
import numpy as np
import pytest

from testing.lang.all_backends import assert_all_backends_match_numpy

RUNTIME_N = "@requires(lambda n, v: 2 <= n <= 6)\ndef kernel(n: Integer, v: Integer):\n"
STATIC = "def kernel(v: Integer):\n"

# Writes through another name after a content fact was stated.
STALE_CONTENT = {
    "dyn sum of zeros": RUNTIME_N + "    x = np.zeros(n, dtype=np.int64)\n    y = x\n    y[0] = v\n    return np.sum(x)\n",
    "dyn sum of ones": RUNTIME_N + "    x = np.ones(n, dtype=np.int64)\n    y = x\n    y[n - 1] = v\n    return np.sum(x)\n",
    "dyn prod of ones": RUNTIME_N + "    x = np.ones(n, dtype=np.int64)\n    y = x\n    y[0] = v\n    return np.prod(x)\n",
    "dyn sum of a concatenation of ones": (
        RUNTIME_N + "    x = np.ones(n, dtype=np.int64)\n    c = np.concatenate([x, x])\n    d = c\n    d[0] = v\n"
        "    return np.sum(c)\n"
    ),
    "dyn concatenation after the source was written": (
        RUNTIME_N + "    x = np.ones(n, dtype=np.int64)\n    y = x\n    y[0] = v\n    c = np.concatenate([x, x])\n"
        "    return np.sum(c)\n"
    ),
    "dyn sum of zeros, write in a loop": (
        RUNTIME_N + "    x = np.zeros(n, dtype=np.int64)\n    for k in range(2):\n        y = x\n        y[k] = v\n"
        "    return np.sum(x)\n"
    ),
    "static sum of zeros": STATIC + "    x = np.zeros(4, dtype=np.int64)\n    y = x\n    y[0] = v\n    return np.sum(x)\n",
    "static prod of ones": STATIC + "    x = np.ones(4, dtype=np.int64)\n    y = x\n    y[1] = v\n    return np.prod(x)\n",
    "static mean of zeros": STATIC + "    x = np.zeros(4)\n    y = x\n    y[2] = v\n    return np.mean(x)\n",
    "static sum and mean of full": STATIC + "    x = np.full(4, 7)\n    y = x\n    y[0] = v\n    return np.sum(x), np.mean(x)\n",
    "static sum of a reshape of zeros": (
        STATIC + "    x = np.zeros(4, dtype=np.int64)\n    r = x.reshape(2, 2)\n    q = r\n    q[0, 0] = v\n"
        "    return np.sum(r), np.sum(x)\n"
    ),
    "where on a zeros condition": STATIC + "    c = np.zeros(3, dtype=bool)\n    d = c\n    d[0] = v > 0\n    return np.where(c, 1, 2)\n",
    "matmul with an identity": STATIC + "    e = np.eye(2)\n    f = e\n    f[0, 1] = v\n    return e @ np.array([[1.0, 2.0], [3.0, 4.0]])\n",
}


@pytest.mark.parametrize("v", [5, 0])
@pytest.mark.parametrize("source", STALE_CONTENT.values(), ids=STALE_CONTENT.keys())
def test_content_fact_lapses_after_a_write(source, v):
    args = (4, v) if source.startswith("@requires") else (v,)
    assert_all_backends_match_numpy(source, *args)
