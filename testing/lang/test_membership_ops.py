import pytest

from zinnia import *
from zinnia.debug.exception import ZinniaException


def _ir_classes(decorated):
    return [
        stmt.get("ir_instance", {}).get("__class__")
        for stmt in ZKCircuit.from_method(decorated).compile().get_ir_stmts()
    ]


def test_in_static_list_uses_equality_or_fold():
    @zk_circuit
    def foo(x: int):
        assert x in [1, 3, 5]

    assert foo(3)
    assert not foo(4)
    classes = _ir_classes(foo)
    assert "EqualIIR" in classes
    assert "LogicalOrIR" in classes


def test_not_in_static_tuple_negates_equality_fold():
    @zk_circuit
    def foo(x: int):
        assert x not in (2, 4)

    assert foo(3)
    assert not foo(4)
    classes = _ir_classes(foo)
    assert "EqualIIR" in classes
    assert "LogicalNotIR" in classes


def test_empty_static_composites_are_constants():
    @zk_circuit
    def empty_list(x: int):
        assert (x in []) == False

    @zk_circuit
    def empty_tuple(x: int):
        assert x not in ()

    assert empty_list(7)
    assert empty_tuple(7)


def test_membership_static_ndarray_flattens_cells():
    @zk_circuit
    def foo(x: int):
        arr = np.array([[1, 2], [3, 4]])
        assert x in arr

    assert foo(3)
    assert not foo(5)


def test_dynamic_ndarray_rhs_rejected_with_membership_diagnostic():
    @zk_circuit
    def foo(x: NDArray[Integer, 4], threshold: int):
        dyn = x[x > threshold]
        assert threshold in dyn

    with pytest.raises(ZinniaException) as exc_info:
        ZKCircuit.from_method(foo).compile()

    msg = str(exc_info.value)
    assert "membership over DynamicNDArray is not supported" in msg
    assert "static List, Tuple, or NDArray RHS" in msg
