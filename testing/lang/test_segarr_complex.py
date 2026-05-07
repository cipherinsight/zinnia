"""Regression tests for `compiler.segarr-complex-dtype` (P5a of the
segment-native-static-arrays epic).

Cover dual-segment Complex `Value::StaticArray` construction, reads, writes,
elementwise / reduction / shape ops, and the `np.real` / `np.imag` /
`np.conj` / `abs` helpers.
"""

from zinnia import *
from zinnia import ZKCircuit


# ───────────────────────── Construction ──────────────────────────────────


def test_complex_zeros_constructor():
    @zk_circuit
    def foo():
        a = np.zeros((4,), dtype=complex)
        assert a[0] == complex(0, 0)
        assert a[3] == complex(0, 0)
    assert foo()


def test_complex_array_literal_construction():
    @zk_circuit
    def foo():
        a = np.array([1 + 2j, 3 + 4j, 5 + 6j])
        assert a[0].real == 1.0
        assert a[0].imag == 2.0
        assert a[2].real == 5.0
        assert a[2].imag == 6.0
    assert foo()


def test_complex_zeros_like_constructor():
    @zk_circuit
    def foo():
        a = np.array([1 + 2j, 3 + 4j])
        b = np.zeros_like(a)
        assert b[0] == complex(0, 0)
        assert b[1] == complex(0, 0)
    assert foo()


def test_complex_ndarray_param_compiles():
    """Complex NDArray param annotation should compile (param parsing now
    routes through dual-segment StaticArray construction). End-to-end
    invocation needs Complex input-parser plumbing which is out of P5a scope.
    """
    @zk_circuit
    def foo(a: NDArray[Complex, 4]):
        # Reference the array shape so the param doesn't get optimised away.
        assert a[0].real == a[0].real
    ZKCircuit.from_method(foo).compile()


# ───────────────────────── Reads ──────────────────────────────────────────


def test_complex_static_index_read():
    @zk_circuit
    def foo():
        a = np.array([1 + 2j, 3 + 4j])
        c = a[1]
        assert c.real == 3.0
        assert c.imag == 4.0
    assert foo()


def test_complex_dynamic_index_read():
    @zk_circuit
    def foo(j: Integer):
        a = np.array([1 + 2j, 3 + 4j, 5 + 6j])
        c = a[j]
        # Sanity assertion that the read returned a Complex with correct components.
        if j == 0:
            assert c.real == 1.0 and c.imag == 2.0
        if j == 1:
            assert c.real == 3.0 and c.imag == 4.0
        if j == 2:
            assert c.real == 5.0 and c.imag == 6.0
    assert foo(0)
    assert foo(1)
    assert foo(2)


# ───────────────────────── Writes ─────────────────────────────────────────


def test_complex_static_setitem():
    @zk_circuit
    def foo():
        a = np.zeros((3,), dtype=complex)
        a[1] = 5 + 6j
        assert a[1].real == 5.0
        assert a[1].imag == 6.0
        # Other cells stay zero.
        assert a[0] == complex(0, 0)
        assert a[2] == complex(0, 0)
    assert foo()


def test_complex_roundtrip_setitem():
    """Pivot-style: copy a Complex from one cell to another via setitem."""
    @zk_circuit
    def foo():
        a = np.array([1 + 2j, 3 + 4j, 5 + 6j])
        b = np.zeros((3,), dtype=complex)
        b[0] = a[2]  # 5+6j
        b[1] = a[0]  # 1+2j
        b[2] = a[1]  # 3+4j
        assert b[0].real == 5.0 and b[0].imag == 6.0
        assert b[1].real == 1.0 and b[1].imag == 2.0
        assert b[2].real == 3.0 and b[2].imag == 4.0
    assert foo()


# ───────────────────────── Elementwise ops ────────────────────────────────


def test_complex_array_addition():
    @zk_circuit
    def foo():
        a = np.array([1 + 2j, 3 + 4j])
        b = np.array([5 + 6j, 7 + 8j])
        c = a + b
        assert c[0].real == 6.0 and c[0].imag == 8.0
        assert c[1].real == 10.0 and c[1].imag == 12.0
    assert foo()


def test_complex_array_scalar_multiplication():
    @zk_circuit
    def foo():
        a = np.array([1 + 2j, 3 + 4j])
        c = a * 2
        assert c[0].real == 2.0 and c[0].imag == 4.0
        assert c[1].real == 6.0 and c[1].imag == 8.0
    assert foo()


def test_complex_array_complex_scalar_multiplication():
    @zk_circuit
    def foo():
        a = np.array([1 + 0j, 0 + 1j])
        c = a * (1 + 1j)
        # (1+0)*(1+i) = 1+i
        # (0+i)*(1+i) = i + i*i = -1 + i
        assert c[0].real == 1.0 and c[0].imag == 1.0
        assert c[1].real == -1.0 and c[1].imag == 1.0
    assert foo()


def test_np_conj_on_complex_array():
    @zk_circuit
    def foo():
        a = np.array([1 + 2j, 3 + 4j])
        c = np.conj(a)
        assert c[0].real == 1.0 and c[0].imag == -2.0
        assert c[1].real == 3.0 and c[1].imag == -4.0
    assert foo()


# ───────────────────────── .real / .imag views ────────────────────────────


def test_real_imag_views_return_floats():
    @zk_circuit
    def foo():
        a = np.array([1 + 2j, 3 + 4j, 5 + 6j])
        r = np.real(a)
        i = np.imag(a)
        assert r[0] == 1.0
        assert r[1] == 3.0
        assert r[2] == 5.0
        assert i[0] == 2.0
        assert i[2] == 6.0
    assert foo()


# ───────────────────────── abs / |z| ──────────────────────────────────────


def test_abs_on_complex_array():
    @zk_circuit
    def foo():
        a = np.array([3 + 4j, 0 + 0j])
        m = np.abs(a)
        # |3+4j| = 5
        assert m[0] == 5.0
        assert m[1] == 0.0
    assert foo()


# ───────────────────────── Reductions ─────────────────────────────────────


def test_np_sum_on_complex_array():
    @zk_circuit
    def foo():
        a = np.array([1 + 2j, 3 + 4j, 5 + 6j])
        s = np.sum(a)
        # 1+3+5 + i*(2+4+6) = 9 + 12j
        assert s.real == 9.0
        assert s.imag == 12.0
    assert foo()


def test_np_prod_on_complex_array():
    @zk_circuit
    def foo():
        a = np.array([1 + 0j, 0 + 1j])
        # 1*(0+i) = 0+i
        p = np.prod(a)
        assert p.real == 0.0
        assert p.imag == 1.0
    assert foo()


# ───────────────────────── Shape ops ──────────────────────────────────────


def test_complex_reshape_metadata():
    @zk_circuit
    def foo():
        a = np.array([1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j])
        b = a.reshape((2, 2))
        assert b[0, 0].real == 1.0
        assert b[0, 1].real == 3.0
        assert b[1, 0].real == 5.0
        assert b[1, 1].imag == 8.0
    assert foo()


def test_complex_concatenate():
    @zk_circuit
    def foo():
        a = np.array([1 + 2j, 3 + 4j])
        b = np.array([5 + 6j])
        c = np.concatenate((a, b))
        assert c[0].real == 1.0
        assert c[1].real == 3.0
        assert c[2].real == 5.0
        assert c[2].imag == 6.0
    assert foo()


def test_complex_transpose_2d():
    @zk_circuit
    def foo():
        a = np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]])
        b = a.transpose()
        # Original a[1,0] = 5+6j; after transpose b[0,1] = 5+6j.
        assert b[0, 1].real == 5.0
        assert b[1, 0].imag == 4.0
    assert foo()
