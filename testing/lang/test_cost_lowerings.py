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


# ── powers with a compile-time exponent ──────────────────────────────────

@pytest.mark.parametrize("body", ["a ** 6", "a ** 0", "a ** 1", "a ** 2.0", "np.power(a, 3)",
                                  "3*a**2 - 2*a**3", "a ** 13"])
def test_constant_power_matches_numpy(body):
    assert_matches_numpy(f'def kernel(a: Float[Array, "4"]):\n    return {body}\n',
                         np.array([0.5, -1.25, 1.1, -0.9]), atol=1e-6)


def test_constant_power_is_square_and_multiply():
    # PowFI computes every power up to its maximum exponent and selects one;
    # a compile-time exponent needs ceil(log2 e) squarings and a product per
    # set bit: a ** 6 is three MulF.
    program = compile_kernel('def kernel(a: Float):\n    return a ** 6\n')
    classes = ir_classes(program)
    assert "PowFIIR" not in classes and classes.count("MulFIR") == 3


def test_runtime_power_keeps_powfi():
    program = compile_kernel('def kernel(a: Float, n: Integer):\n    return a ** n\n')
    assert "PowFIIR" in ir_classes(program)


def test_constant_power_halo2_base_matches_mock():
    assert_backends_agree('def kernel(a: Float[Array, "3"]):\n    return a ** 6\n',
                          np.array([0.5, -1.25, 1.1]))


# ── sums of products rescaled once ───────────────────────────────────────

@pytest.mark.parametrize("body", ["a.std()", "a.var()", "np.var(a, ddof=1)",
                                  "(a.mean() - 3 * a.std(), a.mean() + 3 * a.std())",
                                  "np.dot(a, a[::-1])",
                                  "np.cov(np.vstack((a, a[::-1])))"])
def test_sum_of_products_matches_numpy(body):
    a = np.array([0.5, -1.25, 3.0, 2.2, -0.7, 1.9])
    assert_matches_numpy(f'def kernel(a: Float[Array, "6"]):\n    return {body}\n', a,
                         atol=1e-6)


def test_variance_rescales_once():
    program = compile_kernel('def kernel(a: Float[Array, "6"]):\n    return a.var()\n')
    classes = ir_classes(program)
    assert "MulFIR" not in classes and classes.count("ProductSumFIR") == 1


def test_matmul_halo2_base_matches_mock():
    assert_backends_agree(
        'def kernel(a: Float[Array, "2 3"], b: Float[Array, "3 2"]):\n    return a @ b\n',
        np.array([[0.5, -1.0, 2.0], [1.5, 0.25, -3.0]]),
        np.array([[1.0, 2.0], [-0.5, 0.75], [0.2, -1.1]]))


# ── comparisons as one strict less-than ──────────────────────────────────

def test_complementary_comparisons_share_one_less_than():
    # `x < t`, `x >= t` and `t > x` are one LtF after the rewrite.
    source = ('def kernel(x: Float[Array, "3"], t: Float):\n'
              '    return (x < t) + (x >= t) * 2 + (t > x) * 4 + (t <= x) * 8\n')
    x = np.array([0.5, 1.0, 2.0])
    assert_matches_numpy(source, x, 1.0)
    classes = ir_classes(compile_kernel(source))
    assert classes.count("LessThanFIR") == 3  # one per element
    assert not {"GreaterThanFIR", "GreaterThanOrEqualFIR", "LessThanOrEqualFIR"} & set(classes)


@pytest.mark.parametrize("op", ["<", "<=", ">", ">="])
def test_comparison_on_equal_and_unequal_values(op):
    source = f'def kernel(a: Integer[Array, "5"], b: Integer[Array, "5"]):\n    return a {op} b\n'
    assert_matches_numpy(source, np.array([1, 2, 3, -4, 5]), np.array([1, 3, 2, -4, -5]))


# ── membership tests as one zero test ────────────────────────────────────

@pytest.mark.parametrize("source, args", [
    ('def kernel(a: Integer[Array, "4"], number: Integer):\n    return number in a\n',
     (np.array([3, -7, 11, 0]), -7)),
    ('def kernel(a: Integer[Array, "4"], number: Integer):\n    return number in a\n',
     (np.array([3, -7, 11, 0]), 5)),
    ('def kernel(A: Integer[Array, "8"], B: Integer[Array, "3"]):\n'
     '    return A[np.in1d(A, B)]\n',
     (np.array([1, 5, 2, 9, 5, -3, 0, 2]), np.array([5, 0, -3]))),
    ('def kernel(A: Float[Array, "5"], B: Float[Array, "2"]):\n'
     '    return np.in1d(A, B)\n',
     (np.array([0.5, 1.5, -2.25, 0.5, 3.0]), np.array([0.5, -2.25]))),
])
def test_membership_matches_numpy(source, args):
    expected = np.asarray(numpy_kernel(source)(*[np.copy(a) for a in args]))
    satisfied, outputs = run_mock(compile_kernel(source), *args)
    assert satisfied
    if "A[" in source:  # runtime-shaped: length, dim, padded payload
        assert outputs[0] == len(expected)
        assert outputs[2:2 + len(expected)] == list(expected)
    else:
        assert np.array_equal(np.asarray(outputs, dtype=float).ravel(), expected.astype(float).ravel())


def test_membership_is_one_zero_test():
    program = compile_kernel(
        'def kernel(a: Integer[Array, "4"], number: Integer):\n    return number in a\n')
    classes = ir_classes(program)
    assert "LogicalOrIR" not in classes
    assert classes.count("EqualIIR") == 1 and classes.count("MulIIR") == 3


def test_membership_halo2_base_matches_mock():
    assert_backends_agree(
        'def kernel(A: Integer[Array, "6"], B: Integer[Array, "3"]):\n    return np.in1d(A, B)\n',
        np.array([1, 5, 2, 9, 0, -3]), np.array([5, 0, -3]))


# ── np.insert at a runtime position ──────────────────────────────────────

INSERT_1D = 'def kernel(a: Float[Array, "5"], pos: Integer, v: Integer):\n    return np.insert(a, pos, v)\n'
INSERT_2D = ('def kernel(a: Float[Array, "3 2"], pos: Integer, row: Float[Array, "2"]):\n'
             '    return np.insert(a, pos, row, axis=0)\n')


@pytest.mark.parametrize("pos", [0, 2, 5])
def test_insert_at_runtime_position_matches_numpy(pos):
    a = np.array([0.5, -1.0, 2.25, 3.0, -0.75])
    assert_matches_numpy(INSERT_1D, a, pos, 7)
    rows = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    assert_matches_numpy(INSERT_2D, rows, min(pos, 3), np.array([-1.5, 9.0]))


def test_insert_position_out_of_range_is_rejected():
    satisfied, _ = run_mock(compile_kernel(INSERT_1D), np.zeros(5), 6, 1)
    assert not satisfied


def test_insert_uses_no_comparison_per_slot():
    # [pos <= i] is the running or of the equality tests [pos == k], k <= i.
    assert "LessThanIIR" not in ir_classes(compile_kernel(INSERT_1D))


def test_insert_halo2_base_matches_mock():
    assert_backends_agree(INSERT_1D, np.array([0.5, -1.0, 2.25, 3.0, -0.75]), 3, 7)


# ── runtime-index reads from known cells, one axis at a time ─────────────

GATHER_3D = 'def kernel(a: Float[Array, "3 4 5"], i: Integer, j: Integer):\n    return a[i, :, j]\n'
CUBE = np.arange(60.0).reshape(3, 4, 5) / 4


@pytest.mark.parametrize("i, j", [(0, 0), (2, 4), (-1, -5), (1, -2)])
def test_mixed_runtime_index_read_matches_numpy(i, j):
    assert_matches_numpy(GATHER_3D, CUBE, i, j)


@pytest.mark.parametrize("i, j", [(3, 0), (0, 5), (-4, 0)])
def test_mixed_runtime_index_read_out_of_range_is_rejected(i, j):
    satisfied, _ = run_mock(compile_kernel(GATHER_3D), CUBE, i, j)
    assert not satisfied


def test_mixed_runtime_index_read_selects_one_axis_at_a_time():
    # Each output a[i, r, j] selects along j within row (r, c) planes shared by
    # nothing else, then along i: (5 - 1) + (3 - 1) selects, no conjunction of
    # the two axes' tests per candidate.
    classes = ir_classes(compile_kernel(GATHER_3D))
    assert "LogicalAndIR" not in classes
    assert classes.count("SelectFIR") == 4 * (3 * 4 + 2)


def test_mixed_runtime_index_read_halo2_base_matches_mock():
    assert_backends_agree(GATHER_3D, CUBE, -1, 2)


# ── np.percentile at a runtime q ─────────────────────────────────────────

PERCENTILE = 'def kernel(a: Float[Array, "20"], p: Integer):\n    return np.percentile(a, p)\n'
SAMPLE20 = np.round(np.random.default_rng(1).normal(size=20) * 3, 3)


def test_runtime_percentile_matches_numpy_for_every_q():
    program = compile_kernel(PERCENTILE)
    for p in range(101):
        satisfied, outputs = run_mock(program, SAMPLE20, p)
        assert satisfied
        assert abs(outputs[0] - np.percentile(SAMPLE20, p)) < 1e-6, p


@pytest.mark.parametrize("p", [-1, 101])
def test_runtime_percentile_outside_0_100_has_no_proof(p):
    assert not run_mock(compile_kernel(PERCENTILE), SAMPLE20, p)[0]


@pytest.mark.parametrize("q", [0, 37, 62.5, 100])
def test_constant_percentile_matches_numpy(q):
    assert_matches_numpy(f'def kernel(a: Float[Array, "20"]):\n    return np.percentile(a, {q})\n',
                         SAMPLE20)


def test_integer_array_percentile_matches_numpy():
    source = 'def kernel(a: Integer[Array, "7"], p: Integer):\n    return np.percentile(a, p)\n'
    a = np.array([5, -2, 9, 9, 0, 3, 1])
    for p in (0, 33, 50, 99, 100):
        assert_matches_numpy(source, a, p)


def test_runtime_percentile_interpolates_once():
    # One product and one division, not one interpolation per possible q.
    classes = ir_classes(compile_kernel(PERCENTILE))
    assert classes.count("DivFIR") == 1 and "MulFIR" not in classes
    assert classes.count("EqualIIR") == 19


def test_runtime_percentile_halo2_base_matches_mock():
    assert_backends_agree(PERCENTILE, SAMPLE20, 37)
