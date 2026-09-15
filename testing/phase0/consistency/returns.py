"""(d) Returned values: `acceptance.md` §3.1, row 4.

The delivery set's dimension terms at their $\\Phi$-maximum must lie in the
baseline's `allowed_meta` oracle, and the number of public cells phase 0's
record implies must cover what the baseline exposes.
"""

from __future__ import annotations

from typing import Sequence

import z3

from testing.phase0.baseline import BaselineRecord
from testing.phase0.consistency.records import ArrayEntry, ReturnCheck
from testing.phase0.phi_solver import PhiSolver


def check_returns(
    delivered: Sequence[ArrayEntry],
    baseline: BaselineRecord,
    solver: PhiSolver | None,
) -> list[ReturnCheck]:
    """§3.1, row 4: the delivery set against the `allowed_meta` oracle.

    The oracle pools the size, the rank and every per-axis extent of the
    NumPy ground truth on the DS-1000 input fixture, so membership is the
    relation `audit_bindings.py` itself checks. A *constant* extent that is
    not a member is a mismatch; a non-constant extent evaluates to an
    envelope rather than to the shape of one run, so its row is recorded as
    `envelope` and is not a failure.
    """
    if baseline.allowed_meta is None:
        reason = baseline.allowed_meta_reason or "the oracle is unavailable"
        return [
            ReturnCheck(entry.path, entry.dims_pretty, [], "not_applicable", reason)
            for entry in delivered
        ]
    allowed = set(baseline.allowed_meta)
    checks = []
    for entry in delivered:
        bounds, unknown = _delivery_bounds(entry, solver)
        if unknown:
            checks.append(
                ReturnCheck(entry.path, entry.dims_pretty, bounds, "unknown",
                            "Z3 could not bound every axis")
            )
        elif entry.constant is None:
            checks.append(
                ReturnCheck(entry.path, entry.dims_pretty, bounds, "envelope",
                            "a non-constant extent bounds a family of shapes")
            )
        else:
            missing = [b for b in bounds if b not in allowed]
            checks.append(
                ReturnCheck(
                    entry.path, entry.dims_pretty, bounds,
                    "member" if not missing else "not_member",
                    None if not missing else f"allowed_meta = {sorted(allowed)}",
                )
            )
    return checks


def _delivery_bounds(
    entry: ArrayEntry, solver: PhiSolver | None
) -> tuple[list[int | None], bool]:
    """Each axis of a delivered array at its maximum under $\\Phi\\wedge\\psi$."""
    if entry.constant is not None:
        return list(entry.constant), False
    bounds: list[int | None] = []
    unknown = False
    for node_id in entry.dims:
        if solver is None or solver.exhausted:
            bounds.append(None)
            unknown = True
            continue
        answer = solver.maximize_expr(solver.term(node_id), solver.term(entry.psi))
        bounds.append(answer.value)
        unknown = unknown or answer.verdict != "bounded"
    return bounds, unknown


#: Cells one element of a delivered value occupies in the public instance.
#: A complex value binds its real and imaginary parts separately
#: (`expose_circuit_return_value`'s `Value::Complex` arm).
_FLOAT_CELLS = {"Float": 1, "Complex": 2}


def _payload_gap(
    delivered: Sequence[ArrayEntry], baseline: BaselineRecord
) -> str | None:
    """Why the float-payload comparison says nothing here, or `None`.

    Three cases, each a property of the *baseline's* record rather than a
    shape phase 0 got wrong:

    1. phase 0 delivers a leaf the baseline would bind as an integer, so the
       count on the other side mixes payload with shape metadata;
    2. the baseline binds no return cell at all — the circuit exposes no
       output, which is what a `coverage.md` §9 defect does to a program
       whose body never compiled;
    3. the baseline binds integer cells beside the float ones, so the two
       sides are not counting the same thing. A delivery set that is all
       `Float` against a baseline that binds only integers is the sharpest
       form of this: the two disagree on the *dtype* of the return, which is
       not a question this check is about.
    """
    if baseline.exposure is None:
        return "the baseline record carries no exposure"
    if not delivered:
        return "phase 0's delivery set has no array or scalar leaf"
    other = sorted({entry.dtype for entry in delivered} - set(_FLOAT_CELLS))
    if other:
        return (
            f"phase 0's delivery set has {', '.join(other)} leaves, whose "
            "cells the baseline binds as integers, which also carry the "
            "runtime shape metadata"
        )
    exposure = baseline.exposure
    bound = exposure.public_int + exposure.public_float + exposure.elided_int
    if bound + exposure.elided_float == 0:
        return (
            "the baseline binds no return cell at all, so it states no "
            "envelope; `coverage.md` §9 lists the defects that compile a "
            "circuit to an empty output"
        )
    integers = exposure.public_int + exposure.elided_int
    if integers:
        return (
            f"the baseline binds {integers} integer and {exposure.float_cells} "
            "float cells where phase 0 delivers only Float and Complex leaves; "
            "the two disagree on the dtype of the return, not on its shape"
        )
    return None


def check_return_payload(
    delivered: Sequence[ArrayEntry], baseline: BaselineRecord, solver: PhiSolver | None
) -> ReturnCheck:
    """The float payload the baseline reserved against phase 0's envelope.

    The one envelope the baseline states for a *returned* value is the number
    of cells it binds, and the *float* half of that count is the half without
    an ambiguity: every float cell is payload, while an integer cell may be
    payload or the runtime shape metadata a dynamic return exposes. So the
    check compares `public_float + elided_float` against
    $\\sum_{\\text{leaves}}\\max\\prod_k\\mathrm{dims}[k]$ over $\\Phi$, and
    only where that comparison means something — see :func:`_payload_gap` for
    the three cases where it does not.

    The sum is one query rather than one per leaf, under the conjunction of
    the leaves' execution conditions — which for a delivery set built under
    the circuit's own $\\psi$ is that one condition.
    """
    paths = [entry.path for entry in delivered]
    gap = _payload_gap(delivered, baseline)
    if gap is not None:
        return ReturnCheck("payload", paths, [], "not_applicable", gap)
    reserved = baseline.exposure.float_cells
    detail = f"the baseline binds {reserved} float cells"
    if solver is None or solver.exhausted:
        return ReturnCheck("payload", paths, [], "unknown", detail)
    total = z3.IntVal(0)
    for entry in delivered:
        total = total + _FLOAT_CELLS[entry.dtype] * solver.size_expression(entry.dims)
    psi = solver.conjunction({entry.psi for entry in delivered})
    answer = solver.bound(total, psi, reserved)
    return ReturnCheck(
        "payload", paths, [answer.value], answer.verdict,
        detail if answer.note is None else f"{detail}; {answer.note}",
    )
