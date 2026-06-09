"""Foundation card for ZinniaType::Complex.

Scalar complex parameter annotations should parse. NDArray[Complex, ...] is
intentionally rejected with a clear "deferred" diagnostic — full support is
tracked in compiler.complex-ndarray-ops.
"""
import pytest

from zinnia import zk_circuit, ZKCircuit, NDArray, Complex, np
from zinnia.debug.exception import ZinniaException


def test_scalar_complex_lowercase_annotation_compiles():
    @zk_circuit
    def foo(c: complex):
        pass

    ZKCircuit.from_method(foo).compile()


def test_scalar_complex_capitalized_annotation_compiles():
    @zk_circuit
    def foo(c: Complex):
        pass

    ZKCircuit.from_method(foo).compile()


def test_np_complex_dtype_aliases_resolve():
    @zk_circuit
    def foo():
        _ = np.complex64
        _ = np.complex128
        _ = np.cdouble

    ZKCircuit.from_method(foo).compile()


def test_ndarray_complex_now_supported_after_ndarray_ops_card():
    """The deferred case from this card was lifted by
    compiler.complex-ndarray-ops; keep as a smoke regression."""
    @zk_circuit
    def foo(arr: NDArray[Complex, 4]):
        _ = arr[0]

    ZKCircuit.from_method(foo).compile()


def test_numeric_scalar_real_imag_attributes_compile():
    @zk_circuit
    def foo():
        x = 3
        y = 2.5
        assert x.real == 3
        assert x.imag == 0
        assert y.real == 2.5
        assert y.imag == 0.0

    assert foo()


def test_dynamic_complex_real_imag_conj_views_compile():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([1 + 2j, 3 + 4j]))
        r = a.real
        i = a.imag
        c = a.conjugate()
        assert r[0] == 1.0
        assert i[1] == 4.0
        _ = c[0].imag

    ZKCircuit.from_method(foo).compile()
