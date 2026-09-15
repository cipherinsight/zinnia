"""Consistency: $\\Phi$ answers the baseline's shape questions at least as well.

`revision/phase0/acceptance.md` §3.2. For every parity circuit that phase 0
accepts and the baseline compiles, this harness compares the two records
(`consistency.compare`) and asserts the pass condition:

* no static shape differs — an input descriptor's shape, and the float
  payload the return binds;
* no dynamic bound of phase 0 exceeds the baseline's — per axis of every
  matched envelope, and for the delivery set as a whole;
* every baseline-discharged fact that is representable is discharged from
  $\\Phi$ — which, as `consistency/` explains, is the *exact* resolution of
  a symbolic input axis and nothing else, because the IR keeps no record of
  a fact the baseline proved.

The tightening list, the unmatched arrays and the obligation answers are
outputs, not conditions.

A circuit phase 0 rejects is skipped with the rejection reason, and so is one
the baseline no longer compiles; `coverage.md` §9 is the list of why, and
`revision/phase0/results/coverage.json` is what this harness reads to know
which is which without re-analysing at collection time.

The unit tests at the end of this file exercise the pure comparison functions
on hand-written records: the two matching rules and the bound verdicts, with
no compiler and no corpus in the way.

Run as a test (one case per circuit):

```sh
MEMRUN_MAX=12G .../memrun.sh .venv/bin/python -m pytest testing/phase0/test_consistency.py -q
```

Run as a report (`revision/phase0/results/consistency.json`):

```sh
MEMRUN_MAX=12G .../memrun.sh .venv/bin/python testing/phase0/test_consistency.py
```
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

import pytest

# The package's `__init__` puts the repository root on `sys.path`; that
# cannot run before the package itself is importable, so a module that is
# also a script does it first.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from testing.phase0 import consistency as C  # noqa: E402
from testing.phase0.baseline import (  # noqa: E402
    RESULTS,
    BaselineRecord,
    baseline_record,
    cache_path,
    phase0_record_isolated,
)
from testing.phase0.corpus import Circuit, parity_set  # noqa: E402

CONSISTENCY_JSON = RESULTS / "consistency.json"
COVERAGE_JSON = RESULTS / "coverage.json"


# ---------------------------------------------------------------------------
# Which circuits the comparison applies to
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def coverage_index() -> dict[str, dict[str, Any]]:
    """`coverage.json` by circuit label, or an empty index when it is absent.

    Read on first use, not at import: collection must not depend on whether
    the coverage harness has run, and a session reads the file once.
    """
    if not COVERAGE_JSON.exists():
        return {}
    rows = json.loads(COVERAGE_JSON.read_text(encoding="utf-8"))
    return {f"{row['program']}::{row['function']}": row for row in rows}


def skip_reason(circuit: Circuit) -> str | None:
    """Why this circuit is out of scope for the comparison, if it is."""
    row = coverage_index().get(circuit.label)
    if row is not None and row["status"] != "ok":
        return f"phase 0 does not accept it: {row['reason']}: {row['detail']}"
    path = cache_path(circuit.program, circuit.function)
    if not path.exists():
        return "no baseline record; run `testing/phase0/baseline.py` first"
    record = BaselineRecord.from_json(json.loads(path.read_text(encoding="utf-8")))
    if not record.ok:
        return f"the baseline does not compile it: {record.status}: {record.error_message}"
    return None


def compare_circuit(circuit: Circuit) -> C.CircuitConsistency:
    """The comparison for one circuit, or a row saying why there is none."""
    baseline = baseline_record(circuit)
    if not baseline.ok:
        return C.CircuitConsistency(
            circuit.program, circuit.function, f"baseline_{baseline.status}",
            baseline.error_message,
        )
    phase0, status = phase0_record_isolated(circuit)
    if phase0 is None:
        return C.CircuitConsistency(
            circuit.program, circuit.function, f"phase0_{status['status']}",
            f"{status.get('reason')}: {status.get('detail')}",
        )
    return C.compare(phase0, baseline)


@pytest.mark.parametrize("circuit", parity_set(), ids=lambda c: c.label)
def test_circuit_is_consistent(circuit: Circuit) -> None:
    reason = skip_reason(circuit)
    if reason is not None:
        pytest.skip(reason)
    row = compare_circuit(circuit)
    assert row.status == "ok", f"{circuit.label}: {row.status}: {row.detail}"
    assert row.failures == [], f"{circuit.label}:\n  " + "\n  ".join(row.failures)


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def summarize(rows: list[C.CircuitConsistency]) -> dict[str, Any]:
    """The headline numbers of `acceptance.md` §5's consistency section."""
    compared = [row for row in rows if row.status == "ok"]
    answers: Counter[str] = Counter()
    for row in compared:
        answers.update(result.answer for result in row.obligations)
    static: Counter[str] = Counter()
    dynamic: Counter[str] = Counter()
    for row in compared:
        for match in row.static_arrays:
            static[match.verdict] += match.count
        for match in row.dynamic_arrays:
            dynamic[match.match] += match.count
    axes = Counter(
        axis.verdict
        for row in compared
        for match in row.dynamic_arrays
        for axis in match.axes
    )
    inputs = Counter(check.verdict for row in compared for check in row.inputs)
    returns = Counter(check.verdict for row in compared for check in row.returns)
    return {
        "circuits": len(rows),
        "compared": len(compared),
        "skipped": dict(Counter(row.status for row in rows if row.status != "ok")),
        "failures": sum(len(row.failures) for row in compared),
        "tightenings": sum(len(row.tightenings) for row in compared),
        "return_tightenings": sum(
            1 for row in compared for c in row.returns if c.verdict == "tighter"
        ),
        "input_verdicts": dict(inputs),
        "static_matches": dict(static),
        "dynamic_matches": dict(dynamic),
        "dynamic_axis_verdicts": dict(axes),
        "return_verdicts": dict(returns),
        "obligation_answers": dict(answers),
        "z3_seconds": round(sum(row.seconds.get("z3", 0.0) for row in compared), 1),
    }


def print_summary(rows: list[C.CircuitConsistency], summary: dict[str, Any]) -> None:
    """The console form of :func:`summarize`, plus every failing circuit."""
    print(
        f"parity set: {summary['circuits']} circuits, "
        f"{summary['compared']} compared, "
        f"{summary['circuits'] - summary['compared']} skipped"
    )
    for status, count in sorted(summary["skipped"].items()):
        print(f"  {count:4d}  {status}")
    for name in (
        "input_verdicts",
        "static_matches",
        "dynamic_matches",
        "dynamic_axis_verdicts",
        "return_verdicts",
        "obligation_answers",
    ):
        entries = sorted(summary[name].items(), key=lambda kv: -kv[1])
        rendered = ", ".join(f"{count} {key}" for key, count in entries) or "none"
        print(f"{name.replace('_', ' '):24s} {rendered}")
    print(f"tightenings              {summary['tightenings']} axis, "
          f"{summary['return_tightenings']} return payload")
    print(f"z3 seconds               {summary['z3_seconds']}")
    disagreements = [
        (row.label, check)
        for row in rows
        if row.status == "ok"
        for check in row.returns
        if check.verdict == "not_member"
    ]
    print(f"return shapes outside allowed_meta  {len(disagreements)}")
    for label, check in disagreements:
        print(f"  {label} {check.path}: {check.bounds}; {check.detail}")
    failing = [row for row in rows if row.status == "ok" and row.failures]
    print(f"failures                 {summary['failures']} in {len(failing)} circuits")
    for row in failing:
        for message in row.failures:
            print(f"  {row.label}: {message}")


def run_report() -> int:
    """Compare every parity circuit, print the summary, write the JSON."""
    rows = []
    circuits = parity_set()
    for index, circuit in enumerate(circuits, start=1):
        row = compare_circuit(circuit)
        print(
            f"[{index:3d}/{len(circuits)}] {row.status:24s} "
            f"{len(row.failures):2d} failures  {row.label}",
            flush=True,
        )
        rows.append(row)
    summary = summarize(rows)
    print_summary(rows, summary)
    RESULTS.mkdir(parents=True, exist_ok=True)
    CONSISTENCY_JSON.write_text(
        json.dumps(
            {"summary": summary, "rows": [row.to_json() for row in rows]}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {CONSISTENCY_JSON}")
    return 0 if summary["failures"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(run_report())
