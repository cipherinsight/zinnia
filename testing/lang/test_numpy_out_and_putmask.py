"""NumPy output arguments (audit findings X5, X6).

A ufunc takes its output array positionally after its inputs
(`np.add(x, 1, y)`); this used to be ignored. `np.putmask` repeats `values`
in flat order instead of broadcasting them. Each test accepts NumPy's result
and rejects the old wrong one.
"""
import numpy as np
import pytest

from zinnia import ZKCircuit
from zinnia.debug.exception import ZinniaException
from studies.user_study.validate import checked_source


def compile_checked(source, shape):
    return ZKCircuit.from_source("kernel", checked_source(source, shape)).compile()


def accepts(program, *args):
    return program.prove(*args, backend="mock").proof_bytes_hex == "mock_satisfied"


@pytest.mark.parametrize("call,numpy_result", [
    ("np.add(x, 1, y)", lambda x: x + 1),
    ("np.minimum(x, 2, y)", lambda x: np.minimum(x, 2)),
    ("np.negative(x, y)", lambda x: -x),
    ("np.multiply(x, x, out=y)", lambda x: x * x),
])
def test_ufunc_out_argument(call, numpy_result):
    program = compile_checked(f'''def kernel(x: Integer[Array, "3"]):
    y = np.zeros_like(x)
    {call}
    return y
''', (3,))
    x = np.array([1, 2, 3])
    assert accepts(program, x, numpy_result(x))
    assert not accepts(program, x, np.zeros(3, dtype=int))


def test_ufunc_out_given_twice_is_rejected():
    with pytest.raises(ZinniaException, match="out given both"):
        ZKCircuit.from_source("kernel", '''def kernel(x: Integer[Array, "3"]):
    y = np.zeros_like(x)
    np.add(x, 1, y, out=y)
    return y
''').compile()


def test_putmask_repeats_values_in_flat_order():
    program = ZKCircuit.from_source("kernel", checked_source('''def kernel(x: Integer[Array, "2 3"], v: Integer[Array, "2 1"]):
    y = x.copy()
    np.putmask(y, y > 0, v)
    return y
''', (2, 3))).compile()
    x = np.array([[1, 2, 3], [4, 5, 6]])
    v = np.array([[10], [20]])
    expected = x.copy()
    np.putmask(expected, expected > 0, v)
    assert expected.tolist() == [[10, 20, 10], [20, 10, 20]]
    assert accepts(program, x, v, expected)
    # The broadcast result the old lowering accepted.
    assert not accepts(program, x, v, np.array([[10, 10, 10], [20, 20, 20]]))


def test_putmask_with_partial_mask_and_scalar():
    program = compile_checked('''def kernel(x: Integer[Array, "4"]):
    y = x.copy()
    np.putmask(y, y < 0, 0)
    return y
''', (4,))
    x = np.array([1, -2, 3, -4])
    assert accepts(program, x, np.array([1, 0, 3, 0]))
    assert not accepts(program, x, x)


def test_putmask_mask_size_mismatch_is_rejected():
    with pytest.raises(ZinniaException, match="mask and data must be the same size"):
        ZKCircuit.from_source("kernel", '''def kernel(x: Integer[Array, "4"]):
    y = x.copy()
    np.putmask(y, np.array([True, False]), 0)
    return y
''').compile()
