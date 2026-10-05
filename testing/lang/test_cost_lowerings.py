"""Cheaper lowerings must compute NumPy's result (perf-ops, 2026-10-06).

Each section covers one lowering that replaced a costlier construction. The
tests compare outputs with NumPy position by position on the mock backend,
check that the circuit backends agree with the mock, and pin the IR shape
that makes the lowering cheaper (so a regression to the old construction
fails here, not only in a cost table).
"""
import numpy as np
import pytest

from testing.lang.mock_outputs import assert_matches_numpy, compile_kernel, numpy_kernel, run_mock

BN254_P = 0x30644E72E131A029B85045B68181585D2833E84879B9709143E1F593F0000001
HALO2_PARAMS = {"k": 12, "precision_bits": 32, "lookup_bits": 8}


def ir_classes(program):
    return [s["ir_instance"]["__class__"] for s in program.get_ir_stmts()]


def assert_backends_agree(source, *args):
    """The halo2-base proof's public values equal the mock's, field element
    by field element."""
    program = compile_kernel(source)
    mock = program.prove(*args, backend="mock")
    circuit = program.prove(*args, backend="halo2-base", params=HALO2_PARAMS)
    mock_values = sorted(e for e in mock.public_values)
    by_statement = {int(e.split("=")[0].rsplit("_", 1)[1]): int(e.split("=")[1]) % BN254_P
                    for e in mock_values}
    expected = [by_statement[k] for k in sorted(by_statement)]
    got = [int.from_bytes(bytes.fromhex(h), "little") for h in circuit.public_values]
    assert got == expected


# ── integer x float products without a rescale ───────────────────────────

INT_TIMES_FLOAT = [
    ('def kernel(a: Float[Array, "4"], k: Integer):\n    return k * a\n',
     (np.array([0.5, -1.25, 3.0, 1e-3]), -7)),
    ('def kernel(a: Float[Array, "4"], k: Integer):\n    return a * k\n',
     (np.array([0.5, -1.25, 3.0, 1e-3]), 12)),
    ('def kernel(a: Float[Array, "3"]):\n    return 3 * a - 2.0 * a * a\n',
     (np.array([0.1, -0.7, 2.5]),)),
    ('def kernel(x: Float, k: Integer):\n    return float(k) * x\n', (-0.3, 5)),
    ('def kernel(a: Float[Array, "3"], m: Integer[Array, "3"]):\n    return m * a\n',
     (np.array([0.25, -0.5, 1.75]), np.array([3, -4, 0]))),
    # A recurrence with integer coefficients (DS-1000 0399).
    ('def kernel(A: Float[Array, "6"], a: Integer, b: Integer, c: Integer):\n'
     '    B = np.empty(len(A))\n'
     '    for k in range(0, len(B)):\n'
     '        if k == 0:\n'
     '            B[k] = a*A[k]\n'
     '        elif k == 1:\n'
     '            B[k] = a*A[k] + b*B[k-1]\n'
     '        else:\n'
     '            B[k] = a*A[k] + b*B[k-1] + c*B[k-2]\n'
     '    return B\n',
     (np.array([0.5, -1.0, 0.25, 2.0, -0.125, 1.5]), 2, -3, 1)),
]


@pytest.mark.parametrize("source, args", INT_TIMES_FLOAT)
def test_integer_times_float_matches_numpy(source, args):
    assert_matches_numpy(source, *args)


@pytest.mark.parametrize("source, args", INT_TIMES_FLOAT[:3])
def test_integer_times_float_halo2_base_matches_mock(source, args):
    assert_backends_agree(source, *args)


@pytest.mark.parametrize("source", [s for s, _ in INT_TIMES_FLOAT[:2]] + [INT_TIMES_FLOAT[5][0]])
def test_integer_times_float_has_no_rescale(source):
    # The old lowering cast the integer to Q32 and paid a MulF rescale
    # (remainder and window range checks) per product.
    assert "MulFIR" not in ir_classes(compile_kernel(source))


def test_fractional_constant_keeps_the_rescale():
    program = compile_kernel('def kernel(a: Float[Array, "2"]):\n    return 0.5 * a\n')
    assert "MulFIR" in ir_classes(program)


# ── products over a zero-length axis ─────────────────────────────────────

@pytest.mark.parametrize("source, args", [
    ('def kernel(a: Float[Array, "3"], b: Float[Array, "3"]):\n'
     '    return np.dot(a[:0], b[:0])\n', (np.ones(3), np.ones(3))),
    ('def kernel(a: Integer[Array, "3"], b: Integer[Array, "3"]):\n'
     '    return np.dot(a[:0], b[:0]) + 1\n', (np.ones(3, dtype=int), np.ones(3, dtype=int))),
    ('def kernel(a: Float[Array, "2 3"], b: Float[Array, "3 2"]):\n'
     '    return np.dot(a[:, :0], b[:0, :])\n', (np.ones((2, 3)), np.ones((3, 2)))),
    ('def kernel(a: Float[Array, "3"], b: Float[Array, "3 2"]):\n'
     '    return np.dot(a[:0], b[:0])\n', (np.ones(3), np.ones((3, 2)))),
    # NPBench cholesky: the first inner step is a dot product of empty rows.
    ('def kernel(A: Float[Array, "3 3"]):\n'
     '    for i in range(1, 3):\n'
     '        for j in range(i):\n'
     '            A[i, j] -= np.dot(A[i, :j], A[j, :j])\n'
     '    return A\n', (np.arange(9.0).reshape(3, 3),)),
])
def test_product_over_empty_axis_matches_numpy(source, args):
    assert_matches_numpy(source, *args)


# ── sorting networks ─────────────────────────────────────────────────────

def _with_ties(n, seed):
    rng = np.random.default_rng(seed)
    return np.round(rng.normal(size=n) * 2) / 2  # many equal values


@pytest.mark.parametrize("n", [5, 20, 60])
def test_sort_matches_numpy(n):
    assert_matches_numpy(f'def kernel(a: Float[Array, "{n}"]):\n    return np.sort(a)\n',
                         _with_ties(n, n))


@pytest.mark.parametrize("n", [5, 20, 60])
def test_argsort_is_the_stable_order(n):
    # 5 and 20 use the adjacent-swap network, 60 Batcher's network with the
    # (value, index) tie-break; NumPy's stable sort is the reference.
    source = f'def kernel(a: Float[Array, "{n}"]):\n    return np.argsort(a)\n'
    a = _with_ties(n, n)
    expected = np.argsort(a, kind="stable")
    satisfied, outputs = run_mock(compile_kernel(source), a)
    assert satisfied
    assert outputs == list(expected)


def test_argsort_of_integers_with_ties():
    a = np.array([3, 1, 2, 1, 3, 0, 2, 2] * 8)
    source = 'def kernel(a: Integer[Array, "64"]):\n    return np.argsort(a)[::-1]\n'
    satisfied, outputs = run_mock(compile_kernel(source), a)
    assert satisfied
    assert outputs == list(np.argsort(a, kind="stable")[::-1])


def test_argsort_halo2_base_matches_mock():
    assert_backends_agree('def kernel(a: Float[Array, "40"]):\n    return np.argsort(a)\n',
                          _with_ties(40, 7))


def test_batcher_network_size():
    # 60 keys: Batcher's merge exchange has 505 comparators where the
    # adjacent-swap network has 1,770.
    program = compile_kernel('def kernel(a: Float[Array, "60"]):\n    return np.sort(a)\n')
    assert ir_classes(program).count("LessThanFIR") == 505
