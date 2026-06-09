"""Regression tests for `compiler.try-except-statements`.

`try` / `except` (and `try/finally`, `try/except/else`, PEP 654 `try/except*`)
have no clean ZK lowering and are rejected at transform time with a hint
pointing the user at an explicit `if` guard. See the design note in
#zinnia:d9abc847.
"""
import builtins
import sys

import pytest

from zinnia import *  # noqa: F401,F403 — `from zinnia import *` shadows `exec` / `compile`; use `builtins.exec` below.


# --- Circuits defined at module level so decoration succeeds; the
# --- transformer rejection fires at call time, matching the pattern used by
# --- test_raise_and_nested_def.py.


@zk_circuit
def _circuit_with_try_except(x: int):
    try:
        _zinnia_result = x + 1
    except RuntimeError:
        _zinnia_result = 0


@zk_circuit
def _circuit_with_try_finally(x: int):
    try:
        _zinnia_result = x + 1
    finally:
        _zinnia_result = 0


@zk_circuit
def _circuit_with_try_except_else(x: int):
    try:
        _zinnia_result = x + 1
    except RuntimeError:
        _zinnia_result = 0
    else:
        _zinnia_result = x


def _assert_try_except_hint(msg: str) -> None:
    """The diagnostic must (a) identify `try` / `except` as the offender and
    (b) point at the `if` guard alternative so users have a concrete fix."""
    assert "try" in msg and "except" in msg, f"missing try/except mention, got: {msg}"
    assert "if" in msg and "guard" in msg, f"missing if-guard hint, got: {msg}"


def test_try_except_rejected_with_hint():
    with pytest.raises(Exception) as excinfo:
        _circuit_with_try_except(5)
    _assert_try_except_hint(str(excinfo.value))


def test_try_finally_rejected_with_hint():
    with pytest.raises(Exception) as excinfo:
        _circuit_with_try_finally(5)
    _assert_try_except_hint(str(excinfo.value))


def test_try_except_else_rejected_with_hint():
    with pytest.raises(Exception) as excinfo:
        _circuit_with_try_except_else(5)
    _assert_try_except_hint(str(excinfo.value))


# PEP 654 `try / except*` is Python 3.11+. The source uses `except*` syntax,
# which won't even parse on 3.10, so the test module is built dynamically via
# compile + exec, with the source registered in linecache so the
# `@zk_circuit` decorator's `inspect.getsource()` call can find it.
# (`from zinnia import *` shadows the builtins `compile` / `exec`, so we go
# through the `builtins` module explicitly.)
@pytest.mark.skipif(sys.version_info < (3, 11),
                    reason="PEP 654 try/except* requires Python 3.11+")
def test_try_except_star_rejected_with_hint():
    import linecache

    src = (
        "from zinnia import zk_circuit\n"
        "@zk_circuit\n"
        "def _circuit_with_try_except_star(x: int):\n"
        "    try:\n"
        "        _zinnia_result = x + 1\n"
        "    except* RuntimeError:\n"
        "        _zinnia_result = 0\n"
    )
    fname = "<try_except_star_dynamic>"
    linecache.cache[fname] = (
        len(src), None, src.splitlines(keepends=True), fname,
    )
    code = builtins.compile(src, fname, "exec")
    ns: dict = {}
    builtins.exec(code, ns)
    with pytest.raises(Exception) as excinfo:
        ns["_circuit_with_try_except_star"](5)
    _assert_try_except_hint(str(excinfo.value))
