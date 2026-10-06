"""`x[m] = f(x[m])` with the mask written twice costs linear, not quadratic, IR (npb-gaps).

`x[mask] = x[mask] - L` already lowered to one select per element when `mask`
was one variable: the read `x[mask]` (a compaction) carries a view of its
sources, an elementwise operation keeps it, and the write by the same mask
puts `f(x[i])` back at position `i`. When the program writes the mask
expression twice (`dx[dx > L / 2] = dx[dx > L / 2] - L`, pythran
periodic_dist), each occurrence built its own statements, the write did not
recognize the read's mask, and it fell back to a prefix-count unpack network
(O(n^2) selects) after an O(n^2) compaction: 1.0M IR statements for
periodic_dist. The write now compares the masks by value
(`IRBuilder::same_value`: equal pure statements on equal operands).
"""
import numpy as np
import pytest

from testing.lang.all_backends import assert_all_backends_match_numpy
from testing.lang.mock_outputs import compile_kernel

TWICE = (
    'def kernel(x: Float[Array, "4 4"], L: Float, on: Boolean):\n'
    '    d = x - x.T\n'
    '    if on:\n'
    '        d[d > L / 2] = d[d > L / 2] - L\n'
    '        d[d < -L / 2] = d[d < -L / 2] + L\n'
    '    return d\n'
)
X = np.array([[0.1, 2.5, -3.0, 0.7], [1.9, -0.4, 4.2, -2.2], [0.0, 3.3, -1.1, 2.8], [-3.9, 0.6, 1.4, -0.2]])


@pytest.mark.parametrize("on", [True, False])
def test_mask_written_twice_matches_numpy(on):
    assert_all_backends_match_numpy(TWICE, X, 4.0, on)


def test_mask_written_twice_costs_as_one_mask():
    def ir_size(source):
        return len(compile_kernel(source).get_ir_stmts())

    once = ('def kernel(x: Float[Array, "64"], L: Float):\n'
            '    m = x > L / 2\n    x[m] = x[m] - L\n    return x\n')
    twice = 'def kernel(x: Float[Array, "64"], L: Float):\n    x[x > L / 2] = x[x > L / 2] - L\n    return x\n'
    assert ir_size(twice) == ir_size(once)


def test_different_masks_keep_numpy_semantics():
    """A different mask on the read side is not paired with the write's
    (the values come from another array), so NumPy's packed assignment holds."""
    source = (
        'def kernel(x: Float[Array, "6"], y: Float[Array, "6"]):\n'
        '    x[x > 0] = y[y > 0]\n'
        '    return x\n'
    )
    x = np.array([1.0, -2.0, 3.0, -4.0, 5.0, -6.0])
    y = np.array([-7.0, 8.0, 9.0, -1.0, 10.0, -2.0])
    assert_all_backends_match_numpy(source, x, y)
