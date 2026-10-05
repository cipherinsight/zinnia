"""(e) The checks of $\\Phi$ and the baseline's facts: `acceptance.md` §3.1, row 5.

Every check $\\varphi\\Rightarrow\\mathrm{defcond}$ the record carries is asked
of Z3 — "can this `defcond` be false?" — up to
:data:`~testing.phase0.consistency.records.CHECK_BUDGET` distinct questions per
circuit; the baseline's own facts are listed rather than approximated, for the
reason the package docstring gives under "what cannot be extracted".
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from testing.phase0.baseline import BaselineRecord
from testing.phase0.consistency.inputs import baseline_envelope, describe_dt
from testing.phase0.consistency.records import (
    CHECK_BUDGET,
    CheckResult,
    FactSummary,
    cut,
)
from testing.phase0.phi_solver import Answer, PhiSolver


#: How the query's answer reads as an answer about a check.
_ANSWERS = {"entailed": "cannot_fail", "refutable": "can_fail"}


def distinct_checks(record: dict[str, Any]) -> list[tuple[int, dict[str, Any], int]]:
    """The checks to answer: `(index, constraint, occurrences)`.

    An unrolled loop records the same check once per iteration, and many of
    those are the *same* `(path, defcond)` pair because the term arena is
    hash-consed. Answering one of them answers all, so the list is
    deduplicated and each row carries how many checks it stands for.
    """
    seen: dict[tuple[int, int], int] = {}
    rows: list[list[Any]] = []
    for index, constraint in enumerate(record["constraints"]):
        if constraint["kind"] != "check":
            continue
        signature = (int(constraint["path"]), int(constraint["defcond"]))
        position = seen.get(signature)
        if position is None:
            seen[signature] = len(rows)
            rows.append([index, constraint, 1])
        else:
            rows[position][2] += 1
    return [(int(i), c, int(n)) for i, c, n in rows]


def discharge_checks(
    record: dict[str, Any], solver: PhiSolver | None, budget: int = CHECK_BUDGET
) -> list[CheckResult]:
    """§3.1, row 5: Z3's answer to the checks $\\Phi$ records.

    The rows past the budget are not reported one by one — a circuit can have
    tens of thousands and the report would be the list — but as one
    `not_attempted` row carrying how many checks it stands for.
    """
    results = []
    skipped_distinct = 0
    skipped_total = 0
    skipped_reason = ""
    for position, (index, constraint, occurrences) in enumerate(distinct_checks(record)):
        reason = _skip_reason(solver, position, budget)
        if reason is not None:
            skipped_distinct += 1
            skipped_total += occurrences
            skipped_reason = reason
            continue
        answer = _answer_check(solver, constraint)
        results.append(
            CheckResult(index, constraint["provenance"], cut(constraint["pretty"]),
                        answer.verdict, answer.seconds, occurrences, note=answer.note)
        )
    if skipped_distinct:
        results.append(
            CheckResult(-1, "(aggregate)", f"{skipped_distinct} distinct questions",
                        "not_attempted", 0.0, skipped_total, skipped_distinct,
                        skipped_reason)
        )
    return results


def _skip_reason(solver: PhiSolver | None, position: int, budget: int) -> str | None:
    """Why this check is not attempted, or `None` if it is."""
    if solver is None:
        return "no solver for this circuit"
    if position >= budget:
        return f"past the per-circuit budget of {budget} checks"
    if solver.exhausted:
        return "the circuit's Z3 budget ran out"
    return None


def _answer_check(solver: PhiSolver, constraint: dict[str, Any]) -> Answer:
    """One check's answer; see the module docstring for the regime."""
    answer = solver.entails(
        int(constraint["defcond"]), int(constraint["path"]), facts_only=True
    )
    return replace(answer, verdict=_ANSWERS.get(answer.verdict, answer.verdict))


#: The baseline's own facts that the IR does not carry, with the reason.
UNEXTRACTABLE_FACTS = (
    "a `while` guard proved false: `visit_while` consumes the resolver's answer "
    "and simply stops unrolling, so the IR records a shorter statement list and "
    "no trace of the query",
    "a `reshape` extent proved constant: the resolved number is folded into the "
    "shape of the value, and the query leaves no statement",
    "an index proved in range: a proved bound emits nothing; only an index the "
    "baseline could *not* bound leaves an `Assert*` statement",
)


def compare_facts(record: dict[str, Any], baseline: BaselineRecord) -> FactSummary:
    """§3.1, row 5: the baseline's facts beside phase 0's checks.

    The only fact of the baseline's that survives into an artifact is an
    *exact* resolution of a symbolic input axis: the annotation named a
    symbol and `program_inputs[i].export()["dt"]` came back a concrete
    `NDArray`, which means the SMT run inside `resolve_shape_type` pinned it.
    :func:`compare_inputs` checks phase 0 agrees, by comparing the same axis.
    """
    exact = [
        f"{parameter.name}: {describe_dt(parameter.declared)} resolved to "
        f"{describe_dt(parameter.resolved)}"
        for parameter in baseline.inputs
        if isinstance(parameter.declared, dict)
        and "SymbolicNDArray" in parameter.declared
        and baseline_envelope(parameter.resolved)[0] == "static_array"
    ]
    return FactSummary(
        baseline_enforced=dict(baseline.enforced_checks),
        phase0_checks=sum(1 for c in record["constraints"] if c["kind"] == "check"),
        exact_shape_resolutions=exact,
        unextractable=list(UNEXTRACTABLE_FACTS),
    )
