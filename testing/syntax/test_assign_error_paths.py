"""Error-path coverage for assignment statements (D02 repairs).

- A bare annotation (`y: Integer` with no value) must raise a clear error
  instead of crashing in `visit_expr(None)`.
- An unsupported assignment target (e.g. an attribute) must surface a
  contextual exception carrying a non-empty message *and* a source location,
  rather than a bare ``NotImplementedError`` that resurfaces as
  ``ZinniaException("")``.
"""

import pytest

from zinnia import *
from zinnia.debug.exception import ZinniaException


def test_bare_annotation_raises_clear_error():
    @zk_circuit
    def foo():
        y: Integer
        return 0

    with pytest.raises(ZinniaException) as e:
        ZKCircuit.from_method(foo).compile()
    msg = str(e.value)
    assert msg.strip() != ""
    assert "bare annotation" in msg


def test_unsupported_assign_target_has_contextual_message():
    @zk_circuit
    def foo(x: Public[Integer]):
        x.foo = 3

    with pytest.raises(ZinniaException) as e:
        ZKCircuit.from_method(foo).compile()
    msg = str(e.value)
    # Non-empty, with a location and the offending target kind.
    assert msg.strip() != ""
    assert "line" in msg
    assert "Unsupported assignment target" in msg
