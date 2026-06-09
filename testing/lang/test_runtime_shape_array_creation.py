"""Runtime-shape fill constructor coverage.

First slice for `compiler.runtime-shape-array-creation`: tuple shapes
with static axes plus one bounded runtime int should promote
zeros/ones/empty to DynamicNDArray instead of requiring a fully-static
shape.
"""

import pytest

from zinnia.api.zk_circuit import ZKCircuit


def _compile(source: str, name: str):
    return ZKCircuit.from_source(name, source).compile()


def test_np_zeros_static_runtime_tuple_shape_compiles():
    source = """
@requires(lambda n: 0 <= n <= 10)
def runtime_shape_zeros(n: int):
    arr = np.zeros((3, n), dtype=int)
    _zinnia_result = arr.sum()
"""

    program = _compile(source, "runtime_shape_zeros")
    assert len(program.get_ir_stmts()) > 0


def test_np_ones_static_runtime_tuple_shape_proves():
    source = """
@requires(lambda n: 0 <= n <= 10)
def runtime_shape_ones(n: int):
    arr = np.ones((2, n), dtype=int)
    assert arr.sum() == 2 * n
"""

    proof = ZKCircuit.from_source("runtime_shape_ones", source).prove(4, backend="mock")
    assert proof.proof_bytes_hex == "mock_satisfied"


def test_np_empty_static_runtime_tuple_shape_compiles():
    source = """
@requires(lambda n: 0 <= n <= 10)
def runtime_shape_empty(n: int):
    arr = np.empty((4, n), dtype=int)
    _zinnia_result = arr.sum()
"""

    program = _compile(source, "runtime_shape_empty")
    assert len(program.get_ir_stmts()) > 0


def test_runtime_tuple_shape_without_bound_still_rejects():
    source = """
def unbounded_shape(n: int):
    arr = np.zeros((3, n), dtype=int)
    _zinnia_result = arr.sum()
"""

    with pytest.raises(Exception, match="shape element at axis 1"):
        _compile(source, "unbounded_shape")
