"""Regression tests for compiler.operator-node-lineno-crash.

When user code in a `@zk_circuit` uses an unsupported operator class
(`BitAnd`, `BitOr`, `BitXor`, `LShift`, `RShift`, `Invert`, `In`, `NotIn`,
`Is`, `IsNot`), the transformer must produce a clean user-facing
`ZinniaException` carrying an `UnsupportedOperatorException` diagnostic
rather than crashing with `AttributeError: 'BitAnd' object has no
attribute 'lineno'`.

Operator-class AST nodes have `_attributes = ()` per the CPython grammar
and therefore do NOT carry source-location attributes. The fix is purely
defensive: `get_dbg` falls back to `getattr(node, 'lineno', 0)` etc.

These tests do NOT add support for the operators themselves; they only
guarantee a clean diagnostic. When/if real bitwise/membership-op support
lands, the messages will change but the exception type contract stays
the same.
"""
import pytest

from zinnia import zk_circuit, ZKCircuit
from zinnia.debug.exception import ZinniaException


def test_bitand_produces_clean_diagnostic():
    @zk_circuit
    def foo(n: int):
        assert (n & 1) == 1

    with pytest.raises(ZinniaException) as exc_info:
        ZKCircuit.from_method(foo).compile()

    msg = str(exc_info.value)
    assert "UnsupportedOperatorException" in msg
    assert "BitAnd" in msg
    # The bug surfaced as AttributeError; make sure that's not what we get.
    assert "AttributeError" not in msg
    assert "lineno" not in msg


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


def test_bitor_produces_clean_diagnostic():
    @zk_circuit
    def foo(n: int):
        assert (n | 1) == 1

    with pytest.raises(ZinniaException) as exc_info:
        ZKCircuit.from_method(foo).compile()

    msg = str(exc_info.value)
    assert "UnsupportedOperatorException" in msg
    assert "BitOr" in msg
    assert "AttributeError" not in msg


def test_invert_produces_clean_diagnostic():
    @zk_circuit
    def foo(n: int):
        assert (~n) == 0

    with pytest.raises(ZinniaException) as exc_info:
        ZKCircuit.from_method(foo).compile()

    msg = str(exc_info.value)
    assert "UnsupportedOperatorException" in msg
    assert "Invert" in msg
    assert "AttributeError" not in msg
