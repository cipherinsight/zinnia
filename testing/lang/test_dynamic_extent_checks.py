"""Runtime extents of bounded-dynamic results stay valid (handoff H3).

The IR generator records `0 <= extent <= envelope maximum` for the runtime
extents it reads (`len`, `.shape`, `.size`), and the operators that produce a
bounded-dynamic array must make that true. Two producers did not:

* a runtime `reshape` checked only that the product of the new dimensions is
  the element count, so `(-1) * (-4) == 4` passed (NumPy raises), and
* an element-wise operation on two bounded-dynamic arrays paired slots by
  index and took the left extent, so arrays of different runtime lengths
  (NumPy raises) proved a value.

Each test shows the honest input accepted and every claim for an input NumPy
rejects refused, including the value the old circuit accepted.
"""
import numpy as np

from zinnia import ZKCircuit


def accepts(program, *args):
    return program.prove(*args, backend="mock").proof_bytes_hex == "mock_satisfied"


def compile_source(source):
    return ZKCircuit.from_source("kernel", source).compile()


RESHAPE = (
    'def kernel(a: Integer[Array, "N<=12"], h: Integer, w: Integer, e: Integer):\n'
    "    b = a.reshape(h, w)\n"
    "    assert b.shape[0] == e\n"
)


def test_runtime_reshape_accepts_valid_dimensions():
    program = compile_source(RESHAPE)
    assert accepts(program, np.array([1, 2, 3, 4]), 2, 2, 2)
    assert accepts(program, np.array([1, 2, 3, 4]), 1, 4, 1)
    assert not accepts(program, np.array([1, 2, 3, 4]), 2, 2, 1)


def test_runtime_reshape_rejects_negative_dimensions():
    program = compile_source(RESHAPE)
    for claimed in (-1, 2, 4):  # -1 was accepted before the fix
        assert not accepts(program, np.array([1, 2, 3, 4]), -1, -4, claimed)


def test_runtime_reshape_rejects_dimension_past_envelope():
    # 0 * 13 == 0 elements, but 13 exceeds the axis bound of 12.
    program = compile_source(RESHAPE)
    assert not accepts(program, np.array([], dtype=int), 13, 0, 13)


def test_runtime_reshape_with_inferred_dimension():
    program = compile_source(
        'def kernel(a: Integer[Array, "N<=12"], k: Integer, e: Integer):\n'
        "    b = a.reshape(k, -1)\n"
        "    assert b.shape[1] == e\n"
    )
    assert accepts(program, np.arange(6), 2, 3)
    assert not accepts(program, np.arange(6), 2, 2)
    for claimed in (-6, 0, 6):  # NumPy rejects k = -1 next to -1, and k = 0
        assert not accepts(program, np.arange(6), -1, claimed)
        assert not accepts(program, np.arange(6), 0, claimed)


ADD = (
    'def kernel(a: Integer[Array, "N<=10"], b: Integer[Array, "M<=5"], e: Integer):\n'
    "    c = a + b\n"
    "    assert c.sum() == e\n"
)


def test_elementwise_operation_on_equal_runtime_lengths():
    program = compile_source(ADD)
    assert accepts(program, np.arange(5), np.arange(5), 20)
    assert not accepts(program, np.arange(5), np.arange(5), 21)


def test_elementwise_operation_on_different_runtime_lengths_is_rejected():
    program = compile_source(ADD)
    for claimed in (20, 0, 38):  # 20 was accepted before the fix
        assert not accepts(program, np.arange(8), np.arange(5), claimed)
