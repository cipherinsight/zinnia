"""Regression tests for `compiler.circuit-statement-raise-and-funcdef`."""
import pytest

from zinnia import *


def test_raise_on_unreachable_path_is_silent():
    # `raise` inside a guard whose condition can be false at runtime.
    # When the guard is false, the assertion-False is masked by the path
    # condition, so the circuit succeeds.
    @zk_circuit
    def foo(x: int):
        if x < 0:
            raise ValueError("negative")
        assert x >= 0

    assert foo(5)


def test_raise_on_reachable_path_fails():
    # When the guard fires, the circuit is unsatisfiable.
    @zk_circuit
    def foo(x: int):
        if x < 0:
            raise ValueError("negative")

    result = foo(-3)
    assert not result, "should be unsatisfiable for negative x"


def test_unconditional_raise_fails():
    @zk_circuit
    def foo(x: int):
        raise RuntimeError("always fail")

    with pytest.raises(Exception) as excinfo:
        foo(0)
    assert "provably unsatisfiable" in str(excinfo.value)


def test_nested_def_pure_with_annotations_is_lifted():
    # A nested def with full annotations and no captures gets auto-lifted
    # into a chip and used at the call site.
    @zk_circuit
    def foo(a: int, b: int):
        def add(x: int, y: int) -> int:
            _zinnia_result = x + y

        c = add(a, b)
        assert c == 7

    assert foo(3, 4)


def test_nested_def_without_annotations_is_lifted_and_inferred():
    @zk_circuit
    def foo(a: int):
        def helper(x):
            _zinnia_result = x + 1

        c = helper(a)
        assert c == 6

    assert foo(5)


@zk_circuit
def _circuit_with_closure_helper(a: int):
    captured = a + 1

    def helper(x: int) -> int:
        _zinnia_result = x + captured

    c = helper(a)


def test_nested_def_with_closure_rejected_with_hint():
    with pytest.raises(Exception) as excinfo:
        _circuit_with_closure_helper(5)
    msg = str(excinfo.value)
    assert "captures variables" in msg or "outer function" in msg, f"missing closure hint, got: {msg}"


def test_conditional_branch_join_rebinds_common_name():
    @zk_circuit
    def foo(x: int):
        if x > 0:
            y = 1
        else:
            y = 2
        assert y == 1 or y == 2

    assert foo(5)


def test_conditional_one_sided_binding_still_rejected():
    @zk_circuit
    def foo(x: int):
        if x > 0:
            y = 1
        assert y == 1

    with pytest.raises(Exception) as excinfo:
        foo(5)
    msg = str(excinfo.value)
    assert "Variable `y` not found" in msg or "not definitely bound" in msg
