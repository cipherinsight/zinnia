"""Coverage: phase 0 accepts every program the baseline compiles.

`revision/phase0/acceptance.md` §1: for every circuit of the parity set,
`analyze_shapes` must return a result and not a rejection. Programs outside
the parity set are reported, not counted.

Every circuit is analysed in a child process with an address-space limit and
a timeout (`corpus.analyze_isolated`); this process never calls
`analyze_shapes` itself, so a program that unrolls past the machine's memory
costs one row of the report rather than the run.

Run as a test (one case per circuit):

```sh
.venv/bin/python -m pytest testing/phase0 -q
```

Run as a report (histogram plus `revision/phase0/results/coverage.json`):

```sh
.venv/bin/python testing/phase0/test_coverage.py
```
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pytest

# The package's `__init__` puts the repository root on `sys.path`; that
# cannot run before the package itself is importable, so a module that is
# also a script does it first.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from testing.phase0 import RESULTS  # noqa: E402
from testing.phase0.corpus import (  # noqa: E402
    Circuit,
    analyze_isolated,
    outside_parity_set,
    parity_set,
)
from testing.phase0.excluded import EXCLUDED  # noqa: E402

COVERAGE_JSON = RESULTS / "coverage.json"


@dataclass
class Row:
    """One line of the coverage report."""

    family: str
    program: str
    function: str
    status: str
    reason: str | None
    detail: str | None
    variables: int | None
    constraints: int | None
    terms: int | None
    descriptors: int | None
    obligations: int | None
    #: `record.summary` of Δ, which `report.py`'s record section prints.
    record: dict[str, int] | None


def row_of(circuit: Circuit, answer: dict[str, Any]) -> Row:
    """One report row from the summary `analyze_isolated` returns."""
    return Row(
        family=circuit.family,
        program=circuit.program,
        function=circuit.function,
        status=answer["status"],
        reason=answer.get("reason"),
        detail=answer.get("detail"),
        variables=answer.get("variables"),
        constraints=answer.get("constraints"),
        terms=answer.get("terms"),
        descriptors=answer.get("descriptors"),
        obligations=answer.get("obligations"),
        record=answer.get("record"),
    )


@pytest.mark.parametrize("circuit", parity_set(), ids=lambda c: c.label)
def test_parity_circuit_is_accepted(circuit: Circuit) -> None:
    """Every parity circuit is accepted, except those `coverage.md` §9 excludes.

    An excluded circuit is an expected rejection (`excluded.EXCLUDED`); if
    phase 0 accepts one, the test fails so that the exclusion is revisited.
    """
    answer = analyze_isolated(circuit)
    exclusion = EXCLUDED.get(circuit.label)
    if exclusion is not None:
        assert answer["status"] != "ok", (
            f"{circuit.label} is accepted but listed in excluded.py "
            f"({exclusion.value}); remove the entry"
        )
        pytest.xfail(f"{exclusion.value}: {answer.get('detail')}")
    assert answer["status"] == "ok", (
        f"{circuit.label}: {answer.get('reason')}: {answer.get('detail')}"
    )


def run_report() -> int:
    """Analyze the whole parity set, print the histogram, write the JSON."""
    rows = [row_of(circuit, analyze_isolated(circuit)) for circuit in parity_set()]
    accepted = [row for row in rows if row.status == "ok"]
    rejected = [row for row in rows if row.status != "ok"]

    print(f"parity set: {len(rows)} circuits, "
          f"{len(accepted)} accepted, {len(rejected)} rejected")
    # Anything but `ok` is a rejection for acceptance.md §1: a timeout or a
    # dead child is phase 0 failing to answer, which is not an acceptance.
    per_family = Counter((row.family, row.status == "ok") for row in rows)
    for family in sorted({row.family for row in rows}):
        ok = per_family[(family, True)]
        no = per_family[(family, False)]
        print(f"  {family:10s} {ok:4d} accepted  {no:4d} rejected")
    if rejected:
        print("rejection histogram:")
        for reason, count in Counter(row.reason for row in rejected).most_common():
            print(f"  {count:4d}  {reason}")
        print("rejected programs:")
        for row in rejected:
            print(f"  {row.program}::{row.function}  {row.reason}: {row.detail}")

    outside = outside_parity_set()
    if outside:
        print(f"outside the parity set ({sum(outside.values())} circuits, not counted):")
        for status, count in outside.most_common():
            print(f"  {count:4d}  {status}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    COVERAGE_JSON.write_text(
        json.dumps([asdict(row) for row in rows], indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {COVERAGE_JSON}")
    return 0 if not rejected else 1


if __name__ == "__main__":
    raise SystemExit(run_report())
