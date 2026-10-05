"""Questions, memo, solver time, unknowns (`revision/phase1/acceptance.md`
§2.4; `revision/README.md` §11, row 3: the formula size is independent of N
and the answers take milliseconds).

Per accepted circuit, from `stats` of the phase 1 record and from the
answers of A themselves:

- questions by kind (MAX, MIN, PROVE, CONST, CANFAIL) and by answering
  layer (memo, syntactic, interval, z3);
- the memo hit rate: memo hits over questions asked, hits included;
- Z3 queries, sat / unsat / unknown, and the CANFAIL sweep's queries and
  the constraints it asserted;
- the cone size distribution over the Z3-answered entries of A that went
  through the sliced path (mean, max, 95th percentile; the sweep's answers
  record the constraints its one solver holds, not a cone) and the per-query
  time distribution over every Z3 answer (mean, max, 95th percentile), both
  read from the answers so that the percentile is exact;
- the oracle's total solver time.

```text
.venv/bin/python -m testing.phase1.questions
```
"""

from __future__ import annotations

from typing import Any, Iterable

from testing.phase1.cache import CACHE_DIR, Entry, entries
from testing.phase1.replay import timed_out

KINDS = ("MAX", "MIN", "PROVE", "CONST", "CANFAIL")
LAYERS = ("memo", "syntactic", "interval", "z3")

#: Circuits whose solver time is above this are named in the report.
SLOW_MILLIS = 5000


def percentile(values: list[int], share: float) -> int:
    """The nearest-rank percentile of `values`; 0 when there are none."""
    if not values:
        return 0
    ordered = sorted(values)
    rank = max(1, int(round(share * len(ordered))))
    return ordered[min(rank, len(ordered)) - 1]


def rows(entry: Entry) -> dict[str, Any] | None:
    if not entry.accepted or entry.phase1 is None:
        return None
    stats = entry.phase1["stats"]
    solver = stats["solver"]
    kinds = stats["kinds"]
    by_kind = {kind: dict(kinds.get(kind, {})) for kind in KINDS}
    asked = sum(int(k.get("asked", 0)) for k in by_kind.values())
    memo = sum(int(k.get("memo_hits", 0)) for k in by_kind.values())
    z3_answers = [a for a in entry.phase1["answers"] if a["layer"] == "z3"]
    # The CANFAIL sweep records the constraints its one incremental solver
    # holds, not a sliced cone (`sweep.rs`), so the cone distribution is
    # over the sliced path alone; the query times are over every query.
    sliced = [a for a in z3_answers if a["kind"] != "CANFAIL"]
    cones = [int(a["cone"]["constraints"]) for a in sliced]
    millis = [int(a["millis"]) for a in z3_answers]
    # A CONST "not constant" records its two models under the `unknown`
    # evidence variant (`oracle/mod.rs`), so the variant alone is not a
    # timeout; `replay.timed_out` tells them apart.
    unknowns = [a for a in z3_answers if timed_out(a)]
    no_value = [a for a in z3_answers if a["kind"] == "CONST"
                and a["evidence"].get("why") == "the model gave no integer for the term"]
    return {
        "label": entry.label,
        "family": entry.family,
        "questions": asked,
        "memo_hits": memo,
        "memo_rate": memo / asked if asked else 0.0,
        "by_kind": {kind: int(k.get("asked", 0)) for kind, k in by_kind.items()},
        "by_layer": {
            "memo": memo,
            "syntactic": sum(int(k.get("syntactic", 0)) for k in by_kind.values()),
            "interval": sum(int(k.get("interval", 0)) for k in by_kind.values()),
            "z3": sum(int(k.get("z3", 0)) for k in by_kind.values()),
        },
        "kind_layers": {
            kind: {layer: int(k.get(field, 0))
                   for layer, field in (("memo", "memo_hits"), ("syntactic", "syntactic"),
                                        ("interval", "interval"), ("z3", "z3"))}
            for kind, k in by_kind.items()
        },
        "queries": int(solver["queries"]),
        "sat": int(solver["sat"]),
        "unsat": int(solver["unsat"]),
        "unknown": int(solver["unknown"]),
        "unknown_answers": len(unknowns),
        "const_without_value": len(no_value),
        "millis": int(solver["millis"]),
        "sweep_queries": int(solver.get("sweep_queries", 0)),
        "sweep_millis": int(solver.get("sweep_millis", 0)),
        "sweep_asserted": int(solver.get("sweep_asserted", 0)),
        "phi_constraints": int(solver.get("phi_constraints", 0)),
        "solvers_built": int(solver.get("solvers_built", 0)),
        "solver_reuse": int(solver.get("solver_reuse", 0)),
        "cone_mean": sum(cones) / len(cones) if cones else 0.0,
        "cone_max": max(cones) if cones else 0,
        "cone_p95": percentile(cones, 0.95),
        "query_mean": sum(millis) / len(millis) if millis else 0.0,
        "query_max": max(millis) if millis else 0,
        "query_p95": percentile(millis, 0.95),
        "z3_answers": len(z3_answers),
        "cones": cones,
        "query_millis": millis,
    }


def summary(table: list[dict[str, Any]]) -> dict[str, Any]:
    questions = sum(r["questions"] for r in table)
    memo = sum(r["memo_hits"] for r in table)
    cones = [c for r in table for c in r["cones"]]
    millis = [m for r in table for m in r["query_millis"]]
    slow = sorted((r for r in table if r["millis"] > SLOW_MILLIS), key=lambda r: -r["millis"])
    return {
        "circuits": len(table),
        "questions": questions,
        "memo_hits": memo,
        "memo_rate": memo / questions if questions else 0.0,
        "by_kind": {kind: sum(r["by_kind"][kind] for r in table) for kind in KINDS},
        "by_layer": {layer: sum(r["by_layer"][layer] for r in table) for layer in LAYERS},
        "kind_layers": {
            kind: {layer: sum(r["kind_layers"][kind][layer] for r in table) for layer in LAYERS}
            for kind in KINDS
        },
        "queries": sum(r["queries"] for r in table),
        "sat": sum(r["sat"] for r in table),
        "unsat": sum(r["unsat"] for r in table),
        "unknown": sum(r["unknown"] for r in table),
        "unknown_answers": sum(r["unknown_answers"] for r in table),
        "const_without_value": sum(r["const_without_value"] for r in table),
        "const_without_value_labels": sorted({r["label"] for r in table if r["const_without_value"]}),
        "millis": sum(r["millis"] for r in table),
        "sweep_queries": sum(r["sweep_queries"] for r in table),
        "sweep_millis": sum(r["sweep_millis"] for r in table),
        "sweep_asserted": sum(r["sweep_asserted"] for r in table),
        "phi_constraints": sum(r["phi_constraints"] for r in table),
        "cone_mean": sum(cones) / len(cones) if cones else 0.0,
        "cone_max": max(cones) if cones else 0,
        "cone_p95": percentile(cones, 0.95),
        "query_mean": sum(millis) / len(millis) if millis else 0.0,
        "query_max": max(millis) if millis else 0,
        "query_p95": percentile(millis, 0.95),
        "slow": [(r["label"], r["millis"], r["queries"], r["cone_max"]) for r in slow],
        "without_solver": 1.0 - (sum(r["queries"] for r in table) / questions) if questions else 1.0,
    }


def table(source: Iterable[Entry]) -> list[dict[str, Any]]:
    return [row for row in (rows(entry) for entry in source) if row is not None]


def main() -> int:
    result = table(entries(CACHE_DIR))
    header = (f"{'circuit':<46}{'Q':>7}{'memo%':>7}{'z3':>6}{'unsat':>6}{'unk':>5}"
              f"{'cone~':>7}{'cone^':>7}{'ms~':>7}{'ms^':>6}{'ms95':>6}{'ms':>7}")
    print(header)
    print("-" * len(header))
    for row in sorted(result, key=lambda r: -r["questions"]):
        print(f"{row['label'][:45]:<46}{row['questions']:>7}{row['memo_rate'] * 100:>6.0f}%"
              f"{row['queries']:>6}{row['unsat']:>6}{row['unknown']:>5}{row['cone_mean']:>7.1f}"
              f"{row['cone_max']:>7}{row['query_mean']:>7.1f}{row['query_max']:>6}"
              f"{row['query_p95']:>6}{row['millis']:>7}")
    totals = summary(result)
    print("-" * len(header))
    print(f"{totals['circuits']} circuits, {totals['questions']} questions, memo hit rate "
          f"{totals['memo_rate'] * 100:.0f}%, {totals['queries']} Z3 queries "
          f"({totals['sat']} sat, {totals['unsat']} unsat, {totals['unknown']} unknown), "
          f"cone mean {totals['cone_mean']:.1f} / p95 {totals['cone_p95']} / max "
          f"{totals['cone_max']}, query ms mean {totals['query_mean']:.2f} / p95 "
          f"{totals['query_p95']} / max {totals['query_max']}, {totals['millis']} ms in total; "
          f"sweep {totals['sweep_queries']} queries, {totals['sweep_asserted']} asserted")
    for label, millis, queries, cone in totals["slow"]:
        print(f"  over {SLOW_MILLIS} ms: {label} {millis} ms over {queries} queries, "
              f"largest cone {cone}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
