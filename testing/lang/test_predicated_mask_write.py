"""A boolean-mask write in a branch not taken changes nothing (npb-gaps).

`arr[mask] = v` under a runtime `if` stores `select(c, new, old)` in memory,
but the write also updated the array's compile-time cache of cells
(`static_array_payload`) with the unconditional `new`, and later reads use
that cache. So the circuits (mock, halo2-base, Noir alike) applied the write
on the path where the branch is not taken and proved a value NumPy never
computes (pythran periodic_dist with `periodicX = False`). The cache now
holds the stored value.
"""
import numpy as np
import pytest

from testing.lang.all_backends import assert_all_backends_match_numpy

X = np.array([[0.1, 2.5, -3.0, 0.7], [1.9, -0.4, 4.2, -2.2], [0.0, 3.3, -1.1, 2.8], [-3.9, 0.6, 1.4, -0.2]])


@pytest.mark.parametrize("body", [
    "        d[d > L / 2] = 0.0\n",
    "        d[d > L / 2] -= L\n",
    "        d[d > L / 2] = d[d > L / 2] - L\n",
])
def test_mask_write_in_a_branch_not_taken_changes_nothing(body):
    source = ('def kernel(x: Float[Array, "4 4"], L: Float, on: Boolean):\n'
              '    d = x - x.T\n    if on:\n' + body + '    return d\n')
    for on in (False, True):
        assert_all_backends_match_numpy(source, X, 4.0, on)
