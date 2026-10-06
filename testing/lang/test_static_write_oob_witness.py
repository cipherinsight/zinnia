"""Lang smoke for the static write bounds-discharge regression.

The static-array dynamic-index WRITE path (`a[i] = v` where `a` is a
segment-backed `Value::StaticArray` and `i` is a runtime witness) had
zero bounds discharge prior to this fix: an out-of-range `i` either
silently no-op'd (overlay path) or wrote an unconstrained address
(memory-write fallback), while the READ path discharged everywhere.

These tests pin the post-fix contract:

* An out-of-range dynamic write must refuse the witness (mock + a real
  proving backend agree).
* An in-range negative-index dynamic write wraps per numpy.

At commit 4b69c08 the OOB cases here `satisfied is True` (the write was
a silent no-op, so the post-write read-back still saw the OLD value and
the surrounding assertion held vacuously) — i.e. the prover forged a
witness for an out-of-bounds program. That is the bug this replaces.
"""
import numpy as np
import pytest

from zinnia import requires, zk_circuit
from zinnia.lang.operator import Integer, NDArray


# ───────────────────── out-of-range dynamic write refused ────────────────

def test_static_write_oob_index_refused_mock():
    """`a[i] = v` with `i = 8` on a length-4 static array is out of range.

    The dyn-write chokepoint discharges `0 <= i < 4`; under the lenient
    default this emits a witness assertion the mock prover cannot
    satisfy. (At 4b69c08 the overlay write silently no-op'd and this
    returned `satisfied is True`.)
    """
    # Read back at STATIC cells: a silent-no-op OOB write leaves every
    # cell unchanged, so this assertion would hold vacuously at 4b69c08.
    @zk_circuit
    def foo(j: Integer, v: Integer):
        a = np.asarray([10, 20, 30, 40])
        a[j] = v
        assert a[0] == 10
        assert a[1] == 20
        assert a[2] == 30
        assert a[3] == 40

    # In-range write satisfies only when it does NOT clobber the asserted
    # cell — write to index 2 with its existing value keeps all four true.
    assert foo(2, 30).satisfied is True
    # In-range write that DOES change a cell makes the static read-back
    # assertion fail (proves the write really lands).
    assert foo(2, 999).satisfied is False
    # Out-of-range write is refused by the bounds discharge.
    assert foo(8, 999).satisfied is False
    # Far-out-of-range write is refused.
    assert foo(100, 999).satisfied is False


def test_static_write_oob_negative_index_refused_mock():
    """`i = -5` on a length-4 array normalizes to -1 (< 0) — out of
    range after the numpy `-n <= i < n` normalization."""
    @zk_circuit
    def foo(j: Integer, v: Integer):
        a = np.asarray([10, 20, 30, 40])
        a[j] = v
        assert a[0] == 10
        assert a[1] == 20
        assert a[2] == 30
        assert a[3] == 40

    assert foo(-5, 999).satisfied is False


def test_static_write_2d_oob_index_refused_mock():
    """Multi-axis dynamic write discharges per axis."""
    @zk_circuit
    def foo(r: Integer, c: Integer, v: Integer):
        a = np.asarray([[1, 2, 3], [4, 5, 6]])
        a[r, c] = v
        assert a[0, 0] == 1
        assert a[1, 2] == 6

    # In-range write of an unasserted cell satisfies.
    assert foo(0, 1, 77).satisfied is True
    # Column out of range (axis len 3).
    assert foo(0, 5, 77).satisfied is False
    # Row out of range (axis len 2).
    assert foo(3, 0, 77).satisfied is False


# ───────────────────── negative index write matches numpy ────────────────

def test_static_write_negative_index_wraps_mock():
    """In-range negative index `a[-1] = v` writes the last cell (numpy)."""
    @zk_circuit
    def foo(v: Integer):
        a = np.asarray([1, 2, 3, 4, 5])
        a[-1] = v
        # static negative index resolves at compile time; sanity on the
        # dynamic sibling below.
        assert a[4] == v

    assert foo(100).satisfied is True


def test_static_write_dynamic_negative_index_wraps_mock():
    """Dynamic in-range negative index wraps per numpy and the written
    value is visible at the equivalent positive position."""
    @zk_circuit
    def foo(j: Integer, v: Integer):
        a = np.asarray([1, 2, 3, 4, 5])
        a[j] = v
        assert a[4] == v

    # j = -1 wraps to index 4.
    assert foo(-1, 100).satisfied is True
    # j = -5 wraps to index 0 — should NOT make a[4] == v hold.
    assert foo(-5, 100).satisfied is False


# ─────────────── satisfying @requires discharges Proved (no witness) ──────

def test_static_write_in_range_with_requires_satisfied():
    """A satisfying `@requires` discharges the write bound Proved; the
    program compiles and the assertion holds."""
    @zk_circuit
    @requires(lambda j, v: 0 <= j)
    @requires(lambda j, v: j < 4)
    def foo(j: Integer, v: Integer):
        a = np.asarray([10, 20, 30, 40])
        a[j] = v
        assert a[j] == v

    assert foo(2, 999).satisfied is True


# ───────────────────────── real backend agreement ────────────────────────

def test_static_write_oob_index_refused_backend():
    """The OOB dynamic write is also refused by a real proving backend,
    not only the mock prover."""
    pytest.importorskip("zinnia")

    @zk_circuit
    def foo(j: Integer, v: Integer):
        a = np.asarray([10, 20, 30, 40])
        a[j] = v
        assert a[0] == 10
        assert a[1] == 20
        assert a[2] == 30
        assert a[3] == 40

    from zinnia.api.zk_circuit import ZKCircuit
    prog = ZKCircuit.from_method(foo).compile()

    # In-range write of the existing value verifies.
    r_ok = prog.prove(2, 30, backend="halo2-base")
    assert prog.verify(r_ok) is True

    # Out-of-range write: the bounds-discharge witness assertion cannot be
    # satisfied. Since C18 (always-on witness-satisfaction check), the
    # halo2-base prove path REFUSES the unsatisfiable witness up front rather
    # than emitting an artifact that only `verify` would reject — a
    # strictly stronger contract than the prove-then-verify-false form this
    # test originally pinned (at 4b69c08 the write silently no-op'd and the
    # artifact verified True).
    with pytest.raises(RuntimeError, match="Witness does not satisfy"):
        prog.prove(8, 999, backend="halo2-base")
