"""max/min/argmax/argmin of arrays known to be sorted, with a runtime length.

`np.arange` and `np.linspace` with a runtime length (and an ascending
direction) state that their output is non-decreasing, and the full reductions
of a bounded-dynamic array use that fact (`ops/dyn_ndarray/aggregation.rs`,
`boundary-read-on-sorted`): the minimum is the first element and the maximum
the last. NumPy's argmax is the first position of the maximum, which ties put
before the last position (`np.argmax(np.linspace(c, c, n))` is 0); the
shortcut used to return `n - 1`. The fact also describes the elements only as
they were when it was stated: a write through another name for the array
(`y = x; y[0] = 100`) keeps `x`'s identity but changes its elements, and the
shortcut used to read the stale order.

Every kernel is checked against NumPy on the mock, and halo2-base and Noir must
prove the mock's values.
"""
import numpy as np
import pytest

from testing.lang.all_backends import assert_all_backends_match_numpy

HEAD = "@requires(lambda n: 2 <= n <= 8)\ndef kernel(n: Integer):\n"

KERNELS = {
    # All elements equal: the first maximum and the first minimum are at 0.
    "argmax all equal": "    return np.argmax(np.linspace(2.0, 2.0, n))\n",
    "argmin all equal": "    return np.argmin(np.linspace(2.0, 2.0, n))\n",
    "argmax all equal int": "    return np.argmax(np.linspace(3, 3, n, dtype=np.int64))\n",
    # Ties at the maximum: [0, 0, 1, 1] for n = 4, [0, 0, 0, 0, 0, 1, 1, 1] for n = 8.
    "argmax ties": "    return np.argmax(np.linspace(0, 1.5, n, dtype=np.int64))\n",
    "argmin ties": "    return np.argmin(np.linspace(0, 1.5, n, dtype=np.int64))\n",
    "max ties": "    return np.max(np.linspace(0, 1.5, n, dtype=np.int64))\n",
    "min ties": "    return np.min(np.linspace(0, 1.5, n, dtype=np.int64))\n",
    # Strictly increasing: the maximum is last.
    "argmax arange": "    return np.argmax(np.arange(n))\n",
    "argmax arange from 3 step 2": "    return np.argmax(np.arange(3, n + 3, 2))\n",
    "argmax linspace increasing": "    return np.argmax(np.linspace(0.0, 3.0, n))\n",
    "argmin arange": "    return np.argmin(np.arange(n))\n",
    "max arange": "    return np.max(np.arange(n))\n",
    "min arange": "    return np.min(np.arange(n))\n",
    # Writes through another name after the fact was stated.
    "max after alias write": "    x = np.arange(n)\n    y = x\n    y[0] = 100\n    return np.max(x)\n",
    "argmax after alias write": "    x = np.arange(n)\n    y = x\n    y[0] = 100\n    return np.argmax(x)\n",
    "min after alias write": "    x = np.arange(n)\n    y = x\n    y[n - 1] = -3\n    return np.min(x)\n",
    "argmin after alias write": "    x = np.arange(n)\n    y = x\n    y[0] = 100\n    return np.argmin(x)\n",
    "argmax after alias write in a loop": (
        "    x = np.linspace(0, 1.5, n, dtype=np.int64)\n"
        "    for k in range(1):\n        y = x\n        y[0] = 5\n"
        "    return np.argmax(x)\n"
    ),
}


@pytest.mark.parametrize("n", [2, 5, 8])
@pytest.mark.parametrize("body", KERNELS.values(), ids=KERNELS.keys())
def test_sorted_reduction_matches_numpy(body, n):
    assert_all_backends_match_numpy(HEAD + body, n)
