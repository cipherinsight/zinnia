"""Complex ndarray support: NumberType::Complex, np.zeros(dtype=complex),
indexing, assignment, .real/.imag on subscripted elements.
"""
from zinnia import zk_circuit, ZKCircuit, NDArray, Complex, np


def test_zeros_dtype_complex():
    @zk_circuit
    def foo():
        a = np.zeros((4,), dtype=complex)
        assert a[0] == 0j
        assert a[3] == 0j

    assert foo()


def test_zeros_dtype_np_complex128():
    @zk_circuit
    def foo():
        a = np.zeros((3,), dtype=np.complex128)
        assert a[1].real == 0.0
        assert a[1].imag == 0.0

    assert foo()


def test_assign_complex_element():
    @zk_circuit
    def foo():
        a = np.zeros((4,), dtype=complex)
        a[0] = 1 + 2j
        a[1] = 3 + 4j
        assert a[0].real == 1.0
        assert a[0].imag == 2.0
        assert a[1].real == 3.0
        assert a[1].imag == 4.0

    assert foo()


def test_complex_array_sum_via_loop():
    @zk_circuit
    def foo():
        a = np.zeros((3,), dtype=complex)
        a[0] = 1 + 0j
        a[1] = 0 + 1j
        a[2] = 1 + 1j
        s = a[0] + a[1] + a[2]  # 2 + 2j
        assert s.real == 2.0
        assert s.imag == 2.0

    assert foo()


def test_subscript_dot_real_imag():
    @zk_circuit
    def foo():
        a = np.zeros((2,), dtype=complex)
        a[0] = 5 + 6j
        # `a[0].real` exercises visit_Attribute on a subscript expression
        assert a[0].real == 5.0
        assert a[0].imag == 6.0

    assert foo()


def test_ndarray_complex_param_annotation_compiles():
    @zk_circuit
    def foo(arr: NDArray[Complex, 4]):
        s = arr[0] + arr[1]
        _ = s

    ZKCircuit.from_method(foo).compile()


def test_np_sum_over_complex_array():
    @zk_circuit
    def foo():
        a = np.zeros((3,), dtype=complex)
        a[0] = 1 + 0j
        a[1] = 0 + 1j
        a[2] = 1 + 1j
        s = np.sum(a)
        assert s.real == 2.0
        assert s.imag == 2.0

    assert foo()


def test_np_dot_complex_1d():
    @zk_circuit
    def foo():
        a = np.zeros((3,), dtype=complex)
        b = np.zeros((3,), dtype=complex)
        a[0] = 1 + 0j; a[1] = 2 + 0j; a[2] = 0 + 1j
        b[0] = 1 + 0j; b[1] = 0 + 1j; b[2] = 1 + 0j
        # 1*1 + 2*1j + 1j*1 = 1 + 3j
        d = np.dot(a, b)
        assert d.real == 1.0
        assert d.imag == 3.0

    assert foo()


def test_np_conj_over_complex_array():
    @zk_circuit
    def foo():
        a = np.zeros((2,), dtype=complex)
        a[0] = 1 + 2j
        a[1] = 3 + 4j
        c0 = a[0].conjugate()
        assert c0.real == 1.0
        assert c0.imag == -2.0

    assert foo()
