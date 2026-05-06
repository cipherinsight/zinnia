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


def test_ndarray_complex_is_rejected_with_clear_error():
    @zk_circuit
    def foo(arr: NDArray[Complex, 4]):
        pass

    with pytest.raises(ZinniaException) as exc_info:
        ZKCircuit.from_method(foo).compile()
    msg = str(exc_info.value)
    assert "Complex" in msg
    assert "complex-ndarray-ops" in msg
