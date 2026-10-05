"""The acceptance criteria of `revision/phase1/acceptance.md` §4, one test
each, over the cache.

The three criteria:

1. every accepted circuit passes every invariant of `README.md` §2 on the
   exported JSON (`invariants.py`);
2. no circuit panics or times out, and every rejection carries a reason
   that names a rule, a phase 0 gap or a decision (`invariants.category_of`);
3. no answer replayed by `replay.py` disagrees with the record.

The tests read `<cache>/tables.json`, which `report.py` writes from its one
pass over the cache, so that the report and the test assert the same
numbers; when that file is absent — the report has not been generated for
this cache — the two cheap criteria are computed here from the cache and
the replay is run with a small sample. A failing test names the circuit.

```text
.venv/bin/python -m pytest -q testing/phase1/test_acceptance.py
```

The cache is filled by `python -m testing.phase1.cache --all`; a missing
cache skips the tests rather than failing them, with the command to run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from testing.phase1 import invariants, replay
from testing.phase1.cache import CACHE_DIR, entries, missing
from testing.phase1.report import tables_path

#: Z3 answers replayed per circuit when the report's tables are absent.
FALLBACK_PER_CIRCUIT = 10


@pytest.fixture(scope="module")
def measured() -> dict[str, Any]:
    """The report's tables, or the two cheap tables computed here."""
    cache = CACHE_DIR
    if not cache.exists() or not any(True for _ in entries(cache)):
        pytest.skip(f"no phase 1 cache at {cache}: run "
                    f"`python -m testing.phase1.cache --all` first")
    absent = missing(cache)
    if absent:
        pytest.fail(f"{len(absent)} parity circuits have no cache file, first "
                    f"{absent[0].label}: run `python -m testing.phase1.cache --all`")
    path: Path = tables_path(cache)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    rows_invariants = [invariants.rows(entry) for entry in entries(cache)]
    rows_replay = []
    for entry in entries(cache):
        rows_replay.extend(replay.rows(entry, FALLBACK_PER_CIRCUIT))
    return {
        "tables": {"invariants": rows_invariants, "replay": rows_replay},
        "summaries": {"invariants": invariants.summary(rows_invariants),
                      "replay": replay.summary(rows_replay)},
        "settings": {"per_circuit": FALLBACK_PER_CIRCUIT},
    }


def test_every_accepted_circuit_holds_the_invariants(measured: dict[str, Any]) -> None:
    """`acceptance.md` §4, first criterion."""
    broken = [(row["label"], row["violations"])
              for row in measured["tables"]["invariants"]
              if row["status"] == "ok" and row["violations"]]
    assert not broken, "invariants of README.md §2 broken on: " + "; ".join(
        f"{label}: {violations[0]}" for label, violations in broken)
    rust = [row["label"] for row in measured["tables"]["invariants"]
            if row["status"] == "ok" and row["rust_violations"]]
    assert not rust, f"the Rust checker reports violations on: {', '.join(rust)}"
    accepted = sum(1 for row in measured["tables"]["invariants"] if row["status"] == "ok")
    assert accepted > 0, "no accepted circuit in the cache"


def test_no_circuit_panics_or_times_out_and_every_rejection_has_a_reason(
    measured: dict[str, Any],
) -> None:
    """`acceptance.md` §4, second criterion."""
    failed = [(row["label"], row["status"], row["detail"])
              for row in measured["tables"]["invariants"]
              if row["status"] not in ("ok", "phase1_rejected", "phase0_rejected")]
    assert not failed, "harness failures: " + "; ".join(
        f"{label} ({status}: {detail[:80]})" for label, status, detail in failed)
    unplaced = [(row["label"], row["detail"])
                for row in measured["tables"]["invariants"]
                if row["status"] == "phase1_rejected"
                and (row["category"] not in invariants.CATEGORIES or not row["line"])]
    assert not unplaced, "rejections without a reason in one of the four categories: " + \
        "; ".join(f"{label}: {detail[:80]}" for label, detail in unplaced)


def test_no_replayed_answer_disagrees(measured: dict[str, Any]) -> None:
    """`acceptance.md` §4, third criterion."""
    disagreements = [row for row in measured["tables"]["replay"] if row["outcome"] == "disagree"]
    assert not disagreements, "replay disagreements: " + "; ".join(
        f"{row['label']} answer {row['answer']} ({row['layer']}): {row['detail']}"
        for row in disagreements[:10])
    assert measured["summaries"]["replay"]["replayed"] > 0, "nothing was replayed"
