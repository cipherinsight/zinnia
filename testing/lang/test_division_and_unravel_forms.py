"""Cheaper lowerings that must keep their statements (layer-arith, step 3).

* Noir divides a float by a float-cast integer divisor `n * 2^32` known to be
  non-negative (a sum of `|x|`, a maximum of `|x|`) as the integer division
  by `n` (`divmod_by_positive`), with no sign projection; its remainder bound
  `n - 1 - r >= 0` also proves `n != 0`.
* Noir records ranges through `|x|`, sums and asserted comparisons, and
  emits an asserted `a == b` as one `assert(a == b)`.
* `np.unravel_index` takes the last coordinate without `% dim` (the index is
  checked into `[0, prod(shape))` first).

Each test: NumPy's result is accepted on the mock, halo2-base and Noir; a
wrong claim and inputs outside NumPy's domain are rejected on all three.
"""
import numpy as np
import pytest

from testing.lang.mock_outputs import compile_kernel
from testing.lang.test_dtype_conversions import halo2_accepts, mock_accepts, noir_accepts

Q = 2.0**32


def accepted_everywhere(program, *args) -> bool:
    return mock_accepts(program, *args) and halo2_accepts(program, *args) and noir_accepts(program, *args)


def rejected_everywhere(program, *args) -> bool:
    return not (mock_accepts(program, *args) or halo2_accepts(program, *args) or noir_accepts(program, *args))


def q32(v):
    """The Q32 quotient the circuits prove for an integer x / n: floor(x * 2^32 / n) / 2^32."""
    return np.floor(np.asarray(v, dtype=np.float64) * Q) / Q


ROW_NORMALIZE = {
    "l1": 'def kernel(X: Integer[Array, "2 3"], expected: Float[Array, "2 3"]):\n'
          '    assert np.all(X / np.abs(X).sum(axis=1).reshape(-1, 1) == expected)\n',
    "linf": 'def kernel(X: Integer[Array, "2 3"], expected: Float[Array, "2 3"]):\n'
            '    assert np.all(X / np.abs(X).max(axis=1).reshape(-1, 1) == expected)\n',
}


def exact_quotients(X, divisors):
    return np.array([[np.floor(x * 2**32 / d) / Q for x, d in zip(row, [dv] * len(row))]
                     for row, dv in zip(X.tolist(), divisors)])


@pytest.mark.parametrize("norm", ["l1", "linf"])
def test_division_by_a_non_negative_integer_divisor(norm):
    program = compile_kernel(ROW_NORMALIZE[norm])
    X = np.array([[3, -7, 2], [-1, 0, 5]])
    divisors = np.abs(X).sum(axis=1) if norm == "l1" else np.abs(X).max(axis=1)
    good = exact_quotients(X, divisors.tolist())
    assert np.allclose(good, X / divisors.reshape(-1, 1), atol=1e-9)
    assert accepted_everywhere(program, X, good)
    wrong = good.copy()
    wrong[1, 2] += 1 / Q
    assert rejected_everywhere(program, X, wrong)
    # A zero row: NumPy gives nan, the circuits have no proof.
    zero_row = np.array([[3, -7, 2], [0, 0, 0]])
    claim = good.copy()
    claim[1] = 0.0
    assert rejected_everywhere(program, zero_row, claim)


UNRAVEL = (
    'def kernel(k: Integer, i: Integer, j: Integer, l: Integer):\n'
    '    c = np.unravel_index(k, (4, 3, 5), order="{order}")\n'
    '    assert c[0] == i\n'
    '    assert c[1] == j\n'
    '    assert c[2] == l\n'
)


@pytest.mark.parametrize("order", ["C", "F"])
def test_unravel_index_without_the_last_modulus(order):
    program = compile_kernel(UNRAVEL.format(order=order))
    for k in (0, 7, 31, 59):
        i, j, l = (int(v) for v in np.unravel_index(k, (4, 3, 5), order=order))
        assert accepted_everywhere(program, k, i, j, l), (order, k)
        assert rejected_everywhere(program, k, i, j, l + 1), (order, k)
    # NumPy raises for an index outside [0, 60): no proof, whatever the claim,
    # including the coordinates the unreduced last axis would give for 60.
    for k in (60, -1):
        with pytest.raises(ValueError):
            np.unravel_index(k, (4, 3, 5), order=order)
        for claim in ((0, 0, 0), (3, 2, 4), (4, 0, 0), (0, 0, 5)):
            assert rejected_everywhere(program, k, *claim), (order, k, claim)


# ── Noir native reads at a root checked once (0319's plane gather) ──────

def plane_gather_kernel():
    """0319's kernel: `c` the inverse permutation, `a[c, :, :]`, compiled for
    the Noir target (whose cost rule picks the native reads)."""
    from zinnia import ZKCircuit
    from zinnia.config.zinnia_config import ZinniaConfig

    return ZKCircuit.from_source("kernel", (
        '@requires(lambda a, permutation, expected: is_permutation(permutation))\n'
        'def kernel(a: Float[Array, "8 6 8"], permutation: Integer[Array, "8"], expected: Float[Array, "8 6 8"]):\n'
        '    c = np.empty_like(permutation)\n'
        '    c[permutation] = np.arange(len(permutation))\n'
        '    assert np.all(a[c, :, :] == expected)\n'), config=ZinniaConfig(backend="noir")).compile()


def test_plane_gather_by_a_checked_root():
    """Each plane index is checked once on Noir and every read of the plane
    uses it; NumPy's result is accepted, a wrong plane rejected, and a
    non-permutation (whose `c` NumPy leaves uninitialized) has no proof."""
    program = plane_gather_kernel()
    a = np.linspace(-2, 2, 384).reshape(8, 6, 8)
    perm = np.array([2, 0, 3, 1, 7, 5, 4, 6])
    c = np.empty_like(perm)
    c[perm] = np.arange(8)
    noir = program.emit_structural(a, perm, a[c], backend="noir")
    assert noir.count("array_index::<") == 8, "one check per plane index"
    assert accepted_everywhere(program, a, perm, a[c])
    wrong = a[c].copy()
    wrong[[0, 1]] = wrong[[1, 0]]
    assert rejected_everywhere(program, a, perm, wrong)
    for bad in ([2, 0, 3, 1, 7, 5, 4, 4], [8, 0, 3, 1, 7, 5, 4, 6], [-1, 0, 3, 1, 7, 5, 4, 6]):
        assert rejected_everywhere(program, a, np.array(bad), a[c])


# ── halo2-base: a bounded input loaded inside its bound check ───────────

RAVEL = (
    'def kernel(index: Integer[Array, "5"], expected: Integer):\n'
    '    dims = np.array([8, 9, 8, 8, 8])\n'
    '    assert np.ravel_multi_index(index, dims=dims, order="{order}") == expected\n'
)


@pytest.mark.parametrize("order", ["C", "F"])
def test_bounded_inputs_keep_their_bounds(order):
    """Each index input is assigned inside the gate of its `< dim` check
    (halo2-base); NumPy's flat index is accepted, a wrong one rejected, and
    an index outside its axis (NumPy raises) has no proof."""
    program = compile_kernel(RAVEL.format(order=order))
    dims = (8, 9, 8, 8, 8)
    for index in ([0, 0, 0, 0, 0], [7, 8, 7, 7, 7], [3, 4, 1, 6, 2]):
        flat = int(np.ravel_multi_index(index, dims, order=order))
        assert accepted_everywhere(program, np.array(index), flat), (order, index)
        assert rejected_everywhere(program, np.array(index), flat + 1), (order, index)
    for index in ([8, 0, 0, 0, 0], [0, 9, 0, 0, 0], [0, 0, 0, 0, -1], [0, 0, 32, 0, 0]):
        with pytest.raises(ValueError):
            np.ravel_multi_index(index, dims, order=order)
        wrapped = sum(i * s for i, s in zip(index, np.cumprod((1,) + dims[:-1]) if order == "F"
                                              else np.cumprod((1,) + dims[::-1][:-1])[::-1]))
        for claim in (0, int(wrapped)):
            assert rejected_everywhere(program, np.array(index), claim), (order, index, claim)
