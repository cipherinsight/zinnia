"""(b) and (c): the arrays phase 0 describes against the ones the baseline
allocates (`acceptance.md` §3.1, rows 2 and 3).

A static array is matched by flat size in allocation order; a dynamic one by
parameter name, and then per axis, where $\\max(\\mathrm{dims}[k])$ over
$\\Phi\\wedge\\psi$ must be at most the baseline's declared bound.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

from testing.phase0.baseline import BaselineRecord
from testing.phase0.consistency.records import (
    ArrayEntry,
    AxisBound,
    DynamicMatch,
    StaticMatch,
    UNMATCHED_ROWS,
    axis_by_solver,
)
from testing.phase0.phi_solver import PhiSolver
from testing.phase0.terms import TermDag


def match_static_arrays(
    entries: Sequence[ArrayEntry], baseline: BaselineRecord
) -> list[StaticMatch]:
    """§3.1, row 2: the baseline's memory segments against phase 0's arrays.

    **The matching rule.** The baseline allocates no static array — an
    `NDArray` value is a vector of SSA cells — so its only static allocation
    is an `AllocateMemoryIR` RAM segment, which carries a flat `size` and no
    shape and exists only for an array the program indexes dynamically. The
    comparison is therefore on flat size: each segment, in `stmt_id` order,
    takes the first not-yet-taken phase 0 array descriptor with constant dims
    whose $\\prod_k\\mathrm{dims}[k]$ equals the segment's size, traversal
    order deciding within a size.

    Leftovers on both sides are reported and neither is a failure: most phase
    0 arrays never need RAM at all, and a segment also backs a *dynamic*
    array, whose phase 0 size is a term and not a literal. A leftover segment
    whose size is exactly some dynamic allocation's `max_length` is that
    second case — `expose_dynamic_ndarray_return` reads a dynamic array's
    payload out of a segment of exactly that many cells — and is reported as
    `dynamic_backing` rather than as unmatched.
    """
    available: dict[int, list[ArrayEntry]] = {}
    for entry in entries:
        size = entry.size
        if size is not None:
            available.setdefault(size, []).append(entry)
    envelopes = {a.max_length for a in baseline.dynamic_allocations}
    taken: set[int] = set()
    matches = []
    for allocation in sorted(baseline.static_allocations, key=lambda a: a.stmt_id):
        pool = available.get(allocation.size, [])
        partner = next((e for e in pool if id(e) not in taken), None)
        rendered = f"segment {allocation.segment_id} of {allocation.size} cells"
        if partner is None:
            verdict = (
                "dynamic_backing" if allocation.size in envelopes else "unmatched_baseline"
            )
            matches.append(StaticMatch(allocation.size, rendered, None, verdict))
            continue
        taken.add(id(partner))
        matches.append(
            StaticMatch(
                allocation.size,
                rendered,
                f"{partner.path} {partner.constant}",
                "matched",
            )
        )
    leftovers = [e for e in entries if e.size is not None and id(e) not in taken]
    return matches + _unmatched_static(leftovers)


def _unmatched_static(leftovers: Sequence[ArrayEntry]) -> list[StaticMatch]:
    """The phase 0 arrays no segment claimed, aggregated once there are many.

    An unrolled loop binds one descriptor per iteration, so a row per array
    would make the record the list; past :data:`UNMATCHED_ROWS` the rows are
    grouped by size and carry a count instead.
    """
    if len(leftovers) <= UNMATCHED_ROWS:
        return [
            StaticMatch(e.size, None, f"{e.path} {e.constant}", "unmatched_phase0")
            for e in leftovers
        ]
    grouped = Counter(e.size for e in leftovers)
    return [
        StaticMatch(size, None, f"{count} arrays of this size", "unmatched_phase0", count)
        for size, count in sorted(grouped.items())
    ]


def compare_dynamic_envelopes(
    entries: Sequence[ArrayEntry],
    baseline: BaselineRecord,
    dag: TermDag,
    solver: PhiSolver | None,
) -> list[DynamicMatch]:
    """§3.1, row 3: phase 0's per-axis maximum against the baseline's envelope.

    **The matching rule.** A dynamic allocation that a
    `WitnessDynamicNDArrayMetaIR` ties to a parameter matches the phase 0
    descriptor of that parameter by *name*, which is exact. The rest — an
    array the program promoted itself — has no name in the IR, so it is
    matched by *order*: the remaining allocations in `stmt_id` order against
    the remaining non-constant-dims descriptors of the same rank in traversal
    order. Both kinds are recorded in the row's `match` field, and leftovers
    on both sides are reported rather than guessed at.
    """
    candidates = [entry for entry in entries if entry.constant is None]
    #: First occurrence wins: a parameter's own descriptor is recorded before
    #: the walk starts, so it comes before any later binding of that name.
    by_name: dict[str, ArrayEntry] = {}
    for entry in candidates:
        by_name.setdefault(entry.path, entry)
    taken: set[int] = set()
    matches: list[DynamicMatch] = []
    leftover = []
    for allocation in sorted(baseline.dynamic_allocations, key=lambda a: a.stmt_id):
        entry = by_name.get(allocation.param_name) if allocation.param_name else None
        if entry is None or id(entry) in taken:
            leftover.append(allocation)
            continue
        taken.add(id(entry))
        matches.append(_dynamic_row(allocation, entry, "param", dag, solver))
    matches += _match_by_order(leftover, candidates, taken, dag, solver)
    unclaimed = [entry for entry in candidates if id(entry) not in taken]
    if len(unclaimed) <= UNMATCHED_ROWS:
        return matches + [
            DynamicMatch(None, entry.path, "unmatched_phase0",
                         detail=f"dims {entry.dims_pretty} have no baseline envelope")
            for entry in unclaimed
        ]
    return matches + [
        DynamicMatch(
            None, f"{len(unclaimed)} descriptors, e.g. {unclaimed[0].path}",
            "unmatched_phase0",
            detail="non-constant dims with no baseline envelope",
            count=len(unclaimed),
        )
    ]


def _match_by_order(
    allocations: Sequence[Any],
    candidates: Sequence[ArrayEntry],
    taken: set[int],
    dag: TermDag,
    solver: PhiSolver | None,
) -> list[DynamicMatch]:
    """The allocations no parameter name claimed, paired in traversal order."""
    matches = []
    for allocation in allocations:
        bounds = allocation.max_shape or [allocation.max_length]
        entry = next(
            (e for e in candidates if id(e) not in taken and len(e.dims) == len(bounds)),
            None,
        )
        if entry is None:
            matches.append(
                DynamicMatch(
                    _describe_allocation(allocation), None, "unmatched_baseline",
                    detail="no phase 0 descriptor of this rank has non-constant dims",
                )
            )
            continue
        taken.add(id(entry))
        matches.append(_dynamic_row(allocation, entry, "order", dag, solver))
    return matches


def _describe_allocation(allocation: Any) -> str:
    return (
        f"dyn array {allocation.array_id} max_shape={allocation.max_shape} "
        f"max_length={allocation.max_length}"
        + (f" ({allocation.param_name})" if allocation.param_name else "")
    )


def _dynamic_row(
    allocation: Any,
    entry: ArrayEntry,
    match: str,
    dag: TermDag,
    solver: PhiSolver | None,
) -> DynamicMatch:
    """One matched pair, with an :class:`AxisBound` per axis."""
    bounds = allocation.max_shape or [allocation.max_length]
    if len(bounds) != len(entry.dims):
        return DynamicMatch(
            _describe_allocation(allocation), entry.path, match,
            detail=(
                f"the baseline's envelope has rank {len(bounds)}, "
                f"phase 0's descriptor rank {len(entry.dims)}"
            ),
        )
    axes = [
        axis_by_solver(dag, solver, node_id, entry.psi, bound, axis)
        for axis, (node_id, bound) in enumerate(zip(entry.dims, bounds))
    ]
    return DynamicMatch(_describe_allocation(allocation), entry.path, match, axes)
