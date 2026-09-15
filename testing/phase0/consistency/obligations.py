"""(e) Obligations and the baseline's facts: `acceptance.md` §3.1, row 5.

Every obligation phase 0 records is answered with Z3, up to
:data:`~testing.phase0.consistency.records.OBLIGATION_BUDGET` distinct
questions per circuit; the baseline's own facts are listed rather than
approximated, for the reason the package docstring gives under "what cannot
be extracted".
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from testing.phase0.baseline import BaselineRecord
from testing.phase0.consistency.inputs import baseline_envelope, describe_dt
from testing.phase0.consistency.records import (
    FactSummary,
    ObligationResult,
    OBLIGATION_BUDGET,
    cut,
)
from testing.phase0.phi_solver import Answer, PhiSolver


#: How each query answer reads for each obligation kind.
_ANSWERS = {
    "PROVE": {"entailed": "valid", "refutable": "invalid"},
    "CANFAIL": {"entailed": "cannot_fail", "refutable": "can_fail"},
}


def distinct_obligations(record: dict[str, Any]) -> list[tuple[int, dict[str, Any], int]]:
    """The obligations to answer: `(index, obligation, occurrences)`.

    An unrolled loop records the same question once per iteration, and many
    of those are the *same* `(kind, term, path)` triple because the term arena
    is hash-consed. Answering one of them answers all, so the list is
    deduplicated and each row carries how many obligations it stands for.
    """
    seen: dict[tuple[str, int, int], int] = {}
    rows: list[list[Any]] = []
    for index, obligation in enumerate(record["obligations"]):
        signature = (
            obligation["kind"],
            int(obligation["term"]["id"]),
            int(obligation["path"]),
        )
        position = seen.get(signature)
        if position is None:
            seen[signature] = len(rows)
            rows.append([index, obligation, 1])
        else:
            rows[position][2] += 1
    return [(int(i), o, int(n)) for i, o, n in rows]


def discharge_obligations(
    record: dict[str, Any], solver: PhiSolver | None, budget: int = OBLIGATION_BUDGET
) -> list[ObligationResult]:
    """§3.1, row 5: Z3's answer to the obligations phase 0 recorded.

    The rows past the budget are not reported one by one — a circuit can have
    tens of thousands and the report would be the list — but as one
    `not_attempted` row per kind, carrying how many obligations it stands for.
    """
    results = []
    skipped: dict[str, tuple[int, int, str]] = {}
    for position, (index, obligation, occurrences) in enumerate(
        distinct_obligations(record)
    ):
        kind = obligation["kind"]
        reason = _skip_reason(solver, position, budget)
        if reason is not None:
            distinct, total, _ = skipped.get(kind, (0, 0, reason))
            skipped[kind] = (distinct + 1, total + occurrences, reason)
            continue
        answer = _answer_obligation(solver, kind, obligation)
        results.append(
            ObligationResult(index, kind, obligation["provenance"], cut(obligation["pretty"]),
                             answer.verdict, answer.seconds, occurrences, note=answer.note)
        )
    results += [
        ObligationResult(-1, kind, "(aggregate)", f"{distinct} distinct questions",
                         "not_attempted", 0.0, total, distinct, reason)
        for kind, (distinct, total, reason) in sorted(skipped.items())
    ]
    return results


def _skip_reason(solver: PhiSolver | None, position: int, budget: int) -> str | None:
    """Why this obligation is not attempted, or `None` if it is."""
    if solver is None:
        return "no solver for this circuit"
    if position >= budget:
        return f"past the per-circuit budget of {budget} obligations"
    if solver.exhausted:
        return "the circuit's Z3 budget ran out"
    return None


def _answer_obligation(solver: PhiSolver, kind: str, obligation: dict[str, Any]) -> Answer:
    """One obligation's answer; see the module docstring for the two regimes."""
    term_id = int(obligation["term"]["id"])
    path_id = int(obligation["path"])
    if kind == "CONST":
        return solver.is_constant(term_id, path_id)
    answer = solver.entails(term_id, path_id, facts_only=True)
    verdict = _ANSWERS.get(kind, {}).get(answer.verdict, answer.verdict)
    return replace(answer, verdict=verdict)


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
    """§3.1, row 5: the baseline's facts beside phase 0's obligations.

    The only fact of the baseline's that survives into an artifact is an
    *exact* resolution of a symbolic input axis: the annotation named a
    symbol and `program_inputs[i].export()["dt"]` came back a concrete
    `NDArray`, which means the SMT run inside `resolve_shape_type` pinned it.
    :func:`compare_inputs` checks phase 0 agrees, by comparing the same axis.
    """
    kinds: dict[str, int] = {}
    for obligation in record["obligations"]:
        kinds[obligation["kind"]] = kinds.get(obligation["kind"], 0) + 1
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
        phase0_obligations=kinds,
        exact_shape_resolutions=exact,
        unextractable=list(UNEXTRACTABLE_FACTS),
    )
