"""Regression tests for compiler.operator-node-lineno-crash.

When user code in a `@zk_circuit` uses an unsupported operator class,
the transformer must produce a clean user-facing `ZinniaException`
carrying an `UnsupportedOperatorException` diagnostic rather than
crashing with `AttributeError: 'X' object has no attribute 'lineno'`.

Operator-class AST nodes have `_attributes = ()` per the CPython grammar
and therefore do NOT carry source-location attributes. The fix is purely
defensive: `get_dbg` falls back to `getattr(node, 'lineno', 0)` etc.

After R3, BitAnd/BitOr/BitXor/LShift/RShift/Invert are supported. The
remaining lineno-prone classes are membership/identity comparisons
(`In`, `NotIn`, `Is`, `IsNot`) routed through `get_comp_op_name_from_node`.
"""
import pytest

from zinnia import zk_circuit, ZKCircuit
from zinnia.debug.exception import ZinniaException


def test_notin_produces_clean_diagnostic():
    @zk_circuit
    def foo(n: int):
        assert n not in [1, 2, 3]

    with pytest.raises(ZinniaException) as exc_info:
        ZKCircuit.from_method(foo).compile()

    msg = str(exc_info.value)
    assert "UnsupportedOperatorException" in msg
    assert "NotIn" in msg
    assert "AttributeError" not in msg
    assert "lineno" not in msg


def test_is_produces_clean_diagnostic():
    @zk_circuit
    def foo(n: int):
        assert (n is None) == False

    with pytest.raises(ZinniaException) as exc_info:
        ZKCircuit.from_method(foo).compile()

    msg = str(exc_info.value)
    assert "UnsupportedOperatorException" in msg
    assert "Is" in msg
    assert "AttributeError" not in msg
    assert "lineno" not in msg
