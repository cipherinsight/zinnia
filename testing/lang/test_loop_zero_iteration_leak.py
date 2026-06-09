"""Regression tests for ``compiler.loop-zero-iteration-leak``.

Loop-body names should leak after loops that actually execute, but a name
introduced only inside a statically/provably unreachable loop body is not
definitely bound. The compiler should reject that post-loop load with a
diagnostic that points at definite assignment, not a generic missing name.
"""
import pytest

from zinnia import *  # noqa: F401,F403


def _assert_zero_trip_unbound_message(msg: str, name: str) -> None:
    assert name in msg
    assert "not definitely bound after loop" in msg
    assert "loop body is unreachable" in msg


def test_for_range_zero_body_binding_rejected_with_definite_assignment_error():
    @zk_circuit
    def zero_for():
        for _i in range(0):
            y = _i * 2
        _zinnia_result = y

    with pytest.raises(ZinniaException) as excinfo:
        ZKCircuit.from_method(zero_for).compile()
    _assert_zero_trip_unbound_message(str(excinfo.value), "y")


def test_while_false_body_binding_rejected_with_definite_assignment_error():
    @zk_circuit
    def false_while():
        while False:
            y = 3
        _zinnia_result = y

    with pytest.raises(ZinniaException) as excinfo:
        ZKCircuit.from_method(false_while).compile()
    _assert_zero_trip_unbound_message(str(excinfo.value), "y")


def test_positive_for_loop_body_binding_still_leaks():
    @zk_circuit
    def positive_for():
        for i in range(1):
            y = i + 2
        assert y == 2

    ZKCircuit.from_method(positive_for).compile()

