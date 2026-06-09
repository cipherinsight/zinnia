"""Diagnostics for deliberately unsupported Python expression forms."""

import pytest

from zinnia import ZKCircuit, zk_circuit
from zinnia.debug.exception import ZinniaException


def _compile_error(fn):
    with pytest.raises(ZinniaException) as exc_info:
        ZKCircuit.from_method(fn).compile()
    msg = str(exc_info.value)
    assert "UnsupportedLangFeatureException" in msg
    assert "rule not implemented" not in msg
    return msg


def test_lambda_expression_rejected_with_prescriptive_diagnostic():
    @zk_circuit
    def foo(n: int):
        f = lambda x: x + 1
        assert f(n) == n + 1

    msg = _compile_error(foo)
    assert "Lambda expressions are not supported" in msg
    assert "named helper" in msg
    assert "inline the expression" in msg


def test_set_literal_rejected_with_membership_hint():
    @zk_circuit
    def foo(n: int):
        allowed = {1, 2, 3}
        assert n in allowed

    msg = _compile_error(foo)
    assert "Set literals are not supported" in msg
    assert "no set type" in msg
    assert "list or tuple" in msg
    assert "membership" in msg


def test_dict_literal_rejected_as_mapping_type():
    @zk_circuit
    def foo(n: int):
        table = {1: 2, 3: 4}
        assert table[n] == 2

    msg = _compile_error(foo)
    assert "Dict literals are not supported" in msg
    assert "mapping types are unsupported" in msg
