"""Regression tests for compiler.operator-node-lineno-crash.

When user code in a `@zk_circuit` uses an unsupported operator class,
the transformer must produce a clean user-facing `ZinniaException`
carrying an `UnsupportedOperatorException` diagnostic rather than
crashing with `AttributeError: 'X' object has no attribute 'lineno'`.

Operator-class AST nodes have `_attributes = ()` per the CPython grammar
and therefore do NOT carry source-location attributes. The fix is purely
defensive: `get_dbg` falls back to `getattr(node, 'lineno', 0)` etc.

After R3, BitAnd/BitOr/BitXor/LShift/RShift/Invert are supported, and
`In`/`NotIn` lower through membership support. The remaining unsupported
classes are identity comparisons (`Is`, `IsNot`); the diagnostic names them
and points at the comparison (operator nodes have no location of their own,
so it used to say "line 0").
"""
import pytest

from zinnia import zk_circuit, ZKCircuit
from zinnia.debug.exception import ZinniaException


def test_is_produces_clean_diagnostic():
    @zk_circuit
    def foo(n: int):
        assert (n is None) == False

    with pytest.raises(ZinniaException) as exc_info:
        ZKCircuit.from_method(foo).compile()

    msg = str(exc_info.value)
    assert "UnsupportedOperatorException" in msg
    assert "`is` / `is not`" in msg and "`==` / `!=`" in msg
    assert "line 0" not in msg and "n is None" in msg
    assert "AttributeError" not in msg
    assert "lineno" not in msg
