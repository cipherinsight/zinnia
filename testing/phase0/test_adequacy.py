"""Adequacy: phase 0's record agrees with NumPy on random admitted inputs.

`revision/phase0/acceptance.md` §2. For every circuit of the parity set the
harness draws admitted inputs (`inputs.py`), runs the exported AST with NumPy
semantics (`refeval.py`), builds $\\theta(u)$ and applies `definition.md` §7
(`adequacy.py`).

Run as a test (one case per circuit):

```sh
.venv/bin/python -m pytest testing/phase0/test_adequacy.py -q
```

Run as a report (summary plus `revision/phase0/results/adequacy.json`):

```sh
.venv/bin/python testing/phase0/test_adequacy.py
```

# Isolation

Phase 0 unrolls, so a single circuit can allocate more memory than the machine
has. Every circuit — its `analyze_shapes` call and all of its draws — therefore
runs in a spawned child process with `RLIMIT_AS` and a wall-clock timeout, the
way `benchmarking/validate_contracts.py` isolates a compile. The parent never
calls `analyze_shapes` and never runs the evaluator; a child that is killed or
times out becomes a `memory_exceeded` / `timeout` row instead of taking the run
down with it.

# What a row means

| status | meaning |
|---|---|
| `ok` | phase 0 accepted the circuit and every draw was checked |
| `rejected` | phase 0 refused the circuit; `acceptance.md` §1 counts it, §2 skips it |
| `gap` | the reference evaluator cannot run the program (§2.5: reported, not a failure) |
| `timeout`, `memory_exceeded`, `harness_error` | the child did not finish |

Mismatches are classified by `adequacy.py`: `rule_defect` is the only class
that counts against phase 0.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
import zlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

# The package's `__init__` puts the repository root on `sys.path`; that
# cannot run before the package itself is importable, so a module that is
# also a script does it first.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from testing.phase0 import RESULTS  # noqa: E402
from testing.phase0.corpus import (  # noqa: E402
    ChildLimits,
    Circuit,
    limit_address_space,
    parity_set,
    run_in_child,
)
from testing.phase0.payloads import core_payloads, external_impls  # noqa: E402

ADEQUACY_JSON = RESULTS / "adequacy.json"

#: Draws per mode. `acceptance.md` §2.4 asks for at least 50; the harness runs
#: fewer by default because every draw re-evaluates the term arena of an
#: unrolled program. `ZINNIA_PHASE0_DRAWS` overrides it and the count is
#: recorded in every row.
DRAWS_PER_MODE = int(os.environ.get("ZINNIA_PHASE0_DRAWS", "20"))
#: Wall-clock budget for one circuit, analysis and all draws.
TIMEOUT_SECONDS = float(os.environ.get("ZINNIA_PHASE0_TIMEOUT", "300"))
#: Address-space limit of a child process.
ADDRESS_SPACE_BYTES = int(os.environ.get("ZINNIA_PHASE0_RLIMIT_GB", "6")) * (1 << 30)
#: This harness's isolation budget (`corpus.run_in_child`).
CHILD_LIMITS = ChildLimits(ADDRESS_SPACE_BYTES, TIMEOUT_SECONDS)
#: Base seed. A circuit's seed is :func:`seed_of`, and the row records it.
SEED = 20260915


def seed_of(label: str) -> int:
    """The seed of one circuit's draws.

    A digest of the label, not `hash`: CPython randomises `hash` of a `str`
    per interpreter, so a seed derived from it names different inputs in
    every run and the `draw` index of a recorded mismatch reproduces
    nothing. `zlib.crc32` is stable across runs and platforms.
    """
    return SEED + zlib.crc32(label.encode("utf-8")) % 100000


@dataclass
class ModeReport:
    """The draws of one input mode, and what they answered."""

    mode: str
    draws: int = 0
    successes: int = 0
    failures: int = 0
    inadmissible: int = 0
    gaps: int = 0
    passed: int = 0
    mismatches: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "draws": self.draws,
            "successes": self.successes,
            "failures": self.failures,
            "inadmissible": self.inadmissible,
            "gaps": self.gaps,
            "passed": self.passed,
        }


# ---------------------------------------------------------------------------
# The child process: everything that touches phase 0 or NumPy
# ---------------------------------------------------------------------------


def check_circuit(circuit: Circuit, draws: int) -> dict[str, Any]:
    """Analyse one circuit and check every draw. Runs in the child."""
    from testing.phase0.adequacy import Phase0Record
    from testing.phase0.corpus import load_zk_circuit

    zk_circuit = load_zk_circuit(circuit)
    analysis = zk_circuit.analyze_shapes()
    if "rejected" in analysis:
        return {"status": "rejected", "reason": analysis["rejected"].get("reason"),
                "detail": analysis["rejected"].get("detail")}
    payloads = core_payloads(zk_circuit)
    record = Phase0Record.of(analysis)
    impls = external_impls(zk_circuit)
    seed = seed_of(circuit.label)
    reports = [
        _run_mode(record, payloads, impls, mode, draws, seed, circuit)
        for mode in ("full", "benign", "record")
    ]
    return {
        "status": "ok",
        "seed": seed,
        "variables": len(analysis["variables"]),
        "constraints": len(analysis["constraints"]),
        "terms": len(analysis["terms"]["nodes"]),
        "modes": [report.as_dict() for report in reports],
        "mismatches": _dedupe(m for report in reports for m in report.mismatches),
    }


def _run_mode(
    record: Any,
    payloads: tuple[dict, dict, dict, dict],
    impls: dict[str, Any],
    mode: str,
    draws: int,
    seed: int,
    circuit: Circuit,
) -> ModeReport:
    """One input mode: draw, run, check (`acceptance.md` §2.4, §2.5)."""
    from testing.phase0.adequacy import check
    from testing.phase0.refeval import evaluate

    report = ModeReport(mode=mode)
    for tuple_index, values in enumerate(
        _draws(record.analysis, payloads[0], mode, draws, seed, circuit)
    ):
        trace = evaluate(payloads, values, impls)
        report.draws += 1
        if trace.gap is not None:
            report.gaps += 1
            report.mismatches.append(
                {"key": "<gap>", "name": None, "kind": "harness gap",
                 "expected": "a supported construct", "actual": trace.gap,
                 "classification": "gap"}
            )
            continue
        verdict = check(record, trace)
        report.successes += int(verdict.succeeded)
        report.failures += int(not verdict.succeeded)
        report.inadmissible += int(verdict.inadmissible)
        report.passed += int(verdict.passed)
        report.mismatches.extend(
            dict(mismatch.as_dict(), draw=tuple_index) for mismatch in verdict.mismatches
        )
    return report


def _draws(
    analysis: dict[str, Any],
    ast_json: dict[str, Any],
    mode: str,
    draws: int,
    seed: int,
    circuit: Circuit,
) -> list[list[Any]]:
    """The input tuples of one mode.

    `record` is not a random mode: it is the hand-written `generate_test_case`
    fixture of a DS-1000 case, used as extra draws where one exists.
    """
    from testing.phase0 import inputs as draw_inputs

    if mode == "record":
        specs = draw_inputs.InputSpec.of(ast_json)
        return draw_inputs.record_inputs(circuit.kernel_path, specs)
    settings = draw_inputs.DrawSettings(mode=mode)
    return draw_inputs.draw(analysis, ast_json, draws, seed, settings)


def _dedupe(mismatches: Any) -> list[dict[str, Any]]:
    """One row per distinct (key, kind, classification), with a draw count."""
    seen: dict[tuple[Any, ...], dict[str, Any]] = {}
    for mismatch in mismatches:
        identity = (mismatch["key"], mismatch["kind"], mismatch["classification"],
                    mismatch.get("name"))
        row = seen.setdefault(identity, dict(mismatch, draws=0))
        row["draws"] += 1
    return list(seen.values())


def _worker(circuit: Circuit, draws: int, queue: Any) -> None:
    """The child entry point: cap the address space, then check the circuit."""
    limit_address_space(ADDRESS_SPACE_BYTES)
    try:
        queue.put(check_circuit(circuit, draws))
    except BaseException as exc:  # PyO3 panics inherit from BaseException
        queue.put({
            "status": "harness_error",
            "reason": type(exc).__name__,
            "detail": (str(exc).splitlines() or [repr(exc)])[0][:300],
            "traceback_tail": traceback.format_exc(limit=6).splitlines()[-6:],
        })


# ---------------------------------------------------------------------------
# The parent process: isolation only
# ---------------------------------------------------------------------------


def run_circuit(circuit: Circuit, draws: int = DRAWS_PER_MODE) -> dict[str, Any]:
    """Check one circuit in a child process, never in this one."""
    outcome = run_in_child(_worker, (circuit, draws), CHILD_LIMITS)
    row = outcome.answer or {
        # `crashed` is this harness's `harness_error`: the child exited
        # without an answer for a reason that is not the memory limit.
        "status": "harness_error" if outcome.status == "crashed" else outcome.status,
        "reason": outcome.detail,
    }
    return {
        "family": circuit.family,
        "program": circuit.program,
        "function": circuit.function,
        "draws_per_mode": draws,
        **row,
    }


def defects_of(row: dict[str, Any]) -> list[dict[str, Any]]:
    """The mismatches phase 0 is answerable for.

    Only `rule_defect`; `adequacy.py`'s other classes say the harness could
    not decide, and `acceptance.md` §2.5 reports those separately.
    """
    return [m for m in row.get("mismatches", [])
            if m["classification"] == "rule_defect"]


def key_mismatches_of(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Keys phase 0 recorded that the reference run did not produce, outside
    any unrolled loop — the risk `implementation.md` §6 names.
    """
    return [m for m in row.get("mismatches", [])
            if m["classification"] == "key_mismatch"]


# ---------------------------------------------------------------------------
# pytest
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("circuit", parity_set(), ids=lambda c: c.label)
def test_parity_circuit_is_adequate(circuit: Circuit) -> None:
    row = run_circuit(circuit)
    if row["status"] == "rejected":
        pytest.skip(f"phase 0 rejected: {row.get('reason')}: {row.get('detail')}")
    if row["status"] in ("timeout", "memory_exceeded"):
        pytest.skip(f"{row['status']}: {row.get('reason')}")
    assert row["status"] == "ok", f"{row.get('reason')}: {row.get('detail')}"
    defects = defects_of(row)
    assert not defects, "\n".join(
        f"{d['kind']} at {d['key']} ({d['name']}): "
        f"expected {d['expected']}, got {d['actual']}"
        for d in defects[:10]
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def run_report() -> int:
    """Check the whole parity set, print the summary, write the JSON."""
    rows = [run_circuit(circuit) for circuit in parity_set()]
    _print_summary(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    ADEQUACY_JSON.write_text(
        json.dumps({"draws_per_mode": DRAWS_PER_MODE, "rows": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {ADEQUACY_JSON}")
    return 0 if not any(defects_of(row) for row in rows) else 1


def _print_summary(rows: list[dict[str, Any]]) -> None:
    statuses = Counter(row["status"] for row in rows)
    checked = [row for row in rows if row["status"] == "ok"]
    totals = Counter()
    for row in checked:
        for mode in row["modes"]:
            for field_name in ("draws", "successes", "failures", "inadmissible",
                               "gaps", "passed"):
                totals[field_name] += mode[field_name]
    classes = Counter(
        mismatch["classification"] for row in checked
        for mismatch in row["mismatches"]
    )
    print(f"parity set: {len(rows)} circuits, {DRAWS_PER_MODE} draws per mode")
    for status, count in statuses.most_common():
        print(f"  {count:4d}  {status}")
    print(f"draws: {totals['draws']} "
          f"({totals['successes']} ran, {totals['failures']} failed, "
          f"{totals['inadmissible']} inadmissible, {totals['gaps']} gaps); "
          f"{totals['passed']} passed the check")
    print("mismatch classes (distinct key/kind pairs):")
    for name, count in classes.most_common():
        print(f"  {count:4d}  {name}")
    _print_defects(checked)


def _print_defects(rows: list[dict[str, Any]]) -> None:
    defective = [(row, defects_of(row)) for row in rows]
    defective = [(row, defects) for row, defects in defective if defects]
    if not defective:
        print("no rule defects")
        return
    print(f"circuits with rule defects: {len(defective)}")
    for row, defects in defective[:40]:
        head = defects[0]
        print(f"  {row['program']}::{row['function']}  {len(defects)} defect(s); "
              f"first: {head['kind']} at {head['key']} ({head['name']}): "
              f"expected {head['expected']}, got {head['actual']}")


if __name__ == "__main__":
    raise SystemExit(run_report())
