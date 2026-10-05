"""`revision/phase0/results/REPORT.md`, the report of `acceptance.md` §5.

One generator over the three result files the harnesses write, plus the
baseline cache and `coverage.md` §9:

| Section | Reads |
|---|---|
| coverage | `results/coverage.json` (`test_coverage.py`) |
| adequacy | `results/adequacy.json`, `results/adequacy_hook.json` (`test_adequacy.py`) |
| consistency | `results/consistency.json` (`test_consistency.py`) |
| record | `coverage.json`'s per-circuit summary of $\\Delta$ (`record.summary`) |
| size | `coverage.json`'s $\\lvert V\\rvert$, $\\lvert C\\rvert$ and term counts, against the kernel's source lines and the baseline's IR statement count |
| defects | `coverage.md` §9's table, the parity circuits phase 0 rejects, and the parity circuits the baseline no longer compiles |

A result file that does not exist yet becomes a one-line "not yet produced"
section, so the report can be regenerated at any point of the sweep and
picks the file up as soon as it appears.

Tables only, with a one-line caption each; the prose belongs in
`implementation.md`.

```sh
.venv/bin/python testing/phase0/report.py
```
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Sequence

# The package's `__init__` puts the repository root on `sys.path`; that
# cannot run before the package itself is importable, so a module that is
# also a script does it first.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from testing.phase0.baseline import BaselineRecord, RESULTS, cache_path  # noqa: E402
from testing.phase0.corpus import parity_set  # noqa: E402

COVERAGE_MD = RESULTS.parent / "coverage.md"
REPORT_MD = RESULTS / "REPORT.md"

#: Rows a detail table keeps before it says how many more there are.
DETAIL_ROWS = 30


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def cell(value: Any) -> str:
    """One table cell: a pipe inside a term would end the column."""
    return str(value).replace("|", "\\|").replace("\n", " ")


def table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    """A GitHub-flavoured markdown table, or a note when there is nothing."""
    body = [" | ".join(cell(v) for v in row) for row in rows]
    if not body:
        return "_none_\n"
    head = " | ".join(cell(h) for h in headers)
    rule = " | ".join("---" for _ in headers)
    return "\n".join([f"| {head} |", f"| {rule} |", *[f"| {line} |" for line in body]]) + "\n"


def capped(rows: list[Sequence[Any]]) -> tuple[list[Sequence[Any]], str]:
    """The first :data:`DETAIL_ROWS` rows, and a note about the rest."""
    if len(rows) <= DETAIL_ROWS:
        return rows, ""
    return rows[:DETAIL_ROWS], f"\n_and {len(rows) - DETAIL_ROWS} more rows._\n"


def load(path: Path) -> Any | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def histogram(counter: Counter) -> str:
    return table(["value", "count"], [[k, v] for k, v in counter.most_common()])


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


def section_coverage(rows: list[dict[str, Any]] | None) -> str:
    if rows is None:
        return "_not yet produced: run `testing/phase0/test_coverage.py`._\n"
    families = sorted({row["family"] for row in rows})
    per_family = [
        [
            family,
            sum(1 for r in rows if r["family"] == family),
            sum(1 for r in rows if r["family"] == family and r["status"] == "ok"),
            sum(1 for r in rows if r["family"] == family and r["status"] != "ok"),
        ]
        for family in families
    ]
    per_family.append(
        [
            "**total**",
            len(rows),
            sum(1 for r in rows if r["status"] == "ok"),
            sum(1 for r in rows if r["status"] != "ok"),
        ]
    )
    rejected = [row for row in rows if row["status"] != "ok"]
    detail, note = capped(
        [[r["program"], r["function"], r["reason"], (r["detail"] or "")[:160]]
         for r in rejected]
    )
    return "\n".join(
        [
            "Accepted and rejected per family (`acceptance.md` §1).\n",
            table(["family", "circuits", "accepted", "rejected"], per_family),
            "\nRejection reasons.\n",
            histogram(Counter(row["reason"] for row in rejected)),
            "\nEvery rejected circuit; `coverage.md` §9 says why each is excluded.\n",
            table(["program", "function", "reason", "detail"], detail) + note,
        ]
    )


# ---------------------------------------------------------------------------
# Adequacy
# ---------------------------------------------------------------------------


def section_adequacy(payload: dict[str, Any] | None, hook: dict[str, Any] | None) -> str:
    if payload is None:
        return "_not yet produced: run `testing/phase0/test_adequacy.py`._\n"
    rows = payload["rows"]
    families = sorted({row["family"] for row in rows})
    statuses = sorted({row["status"] for row in rows})
    per_family = [
        [family]
        + [sum(1 for r in rows if r["family"] == family and r["status"] == s)
           for s in statuses]
        for family in families
    ]
    per_family.append(
        ["**total**"] + [sum(1 for r in rows if r["status"] == s) for s in statuses]
    )
    modes = _adequacy_modes(rows)
    mismatches = Counter(
        m["classification"] for row in rows for m in row.get("mismatches", [])
    )
    defects = sorted(
        {
            row["program"] + "::" + row["function"]
            for row in rows
            if any(m["classification"] == "rule_defect" for m in row.get("mismatches", []))
        }
    )
    return "\n".join(
        [
            f"Per family, {payload['draws_per_mode']} draws per input mode "
            "(`acceptance.md` §2.5).\n",
            table(["family", *statuses], per_family),
            "\nDraws per input mode; a failing run tests G3 only.\n",
            table(
                ["mode", "draws", "successes", "failures", "harness gaps", "passed"],
                modes,
            ),
            "\nMismatch classification; every class but `rule_defect` is a "
            "harness question — `unrolled_beyond_run`, `unresolved_variable`, "
            "`numpy_limit`, `ill_typed`.\n",
            histogram(mismatches),
            f"\nThe {len(defects)} circuits with a rule defect; "
            "`implementation.md` names each defect.\n",
            table(["circuit"], [[name] for name in defects]),
            "\nSecond corpus, the suites under `testing/` (`acceptance.md` §2.6).\n",
            _hook_table(hook),
        ]
    )


def _adequacy_modes(rows: list[dict[str, Any]]) -> list[list[Any]]:
    """Totals per input mode over every circuit that ran any draw."""
    totals: dict[str, Counter[str]] = {}
    for row in rows:
        for mode in row.get("modes", ()):
            counts = totals.setdefault(mode["mode"], Counter())
            for key in ("draws", "successes", "failures", "gaps", "passed"):
                counts[key] += mode.get(key, 0)
    return [
        [name, c["draws"], c["successes"], c["failures"], c["gaps"], c["passed"]]
        for name, c in totals.items()
    ]


def _hook_table(hook: dict[str, Any] | None) -> str:
    if hook is None:
        return "_not yet produced: run the suites with `ZINNIA_PHASE0_ADEQUACY=1`._\n"
    return table(
        ["executions", "checked", "passed", "harness gaps", "mismatch classes"],
        [
            [
                hook.get("executions", 0),
                hook.get("checked", 0),
                hook.get("passed", 0),
                sum(hook.get("gaps", {}).values()),
                ", ".join(f"{k}: {v}" for k, v in hook.get("mismatch_classes", {}).items())
                or "none",
            ]
        ],
    )


# ---------------------------------------------------------------------------
# Consistency
# ---------------------------------------------------------------------------


def section_consistency(payload: dict[str, Any] | None) -> str:
    if payload is None:
        return "_not yet produced: run `testing/phase0/test_consistency.py`._\n"
    summary, rows = payload["summary"], payload["rows"]
    compared = [row for row in rows if row["status"] == "ok"]
    return "\n".join(
        [
            "How far the comparison reached (`acceptance.md` §3.2).\n",
            table(
                ["circuits", "compared", "skipped", "failures",
                 "axis tightenings", "payload tightenings", "Z3 s"],
                [
                    [
                        summary["circuits"],
                        summary["compared"],
                        summary["circuits"] - summary["compared"],
                        summary["failures"],
                        summary["tightenings"],
                        summary["return_tightenings"],
                        summary["z3_seconds"],
                    ]
                ],
            ),
            "\nWhy a circuit was skipped.\n",
            histogram(Counter(summary["skipped"])),
            "\nThe five relations of §3.1, by verdict.\n",
            _verdict_table(summary),
            "\nTightenings: a bound phase 0 puts strictly below the baseline's.\n",
            _tightening_table(compared),
            "\nUnmatched arrays, both sides.\n",
            _unmatched_table(compared),
            "\nChecks, by answer; `not_attempted` is the per-circuit budget.\n",
            _check_table(compared),
            "\nThe baseline's enforced checks against phase 0's.\n",
            _facts_table(compared),
            "\nDelivered shapes outside the `allowed_meta` oracle: a "
            "disagreement with NumPy on the DS-1000 fixture, which "
            "`acceptance.md` §3.2 does not make a pass condition.\n",
            _oracle_table(compared),
            "\nViolations of the pass condition.\n",
            table(
                ["circuit", "violation"],
                [[row["program"] + "::" + row["function"], message]
                 for row in compared
                 for message in _failures_of(row)],
            ),
        ]
    )


def _oracle_table(rows: list[dict[str, Any]]) -> str:
    """Every delivered shape the `allowed_meta` oracle does not admit."""
    return table(
        ["circuit", "leaf", "phase 0 dims at their bound", "allowed_meta"],
        [
            [row["program"] + "::" + row["function"], check["path"],
             check["bounds"], check["detail"]]
            for row in rows
            for check in row["returns"]
            if check["verdict"] == "not_member"
        ],
    )


def _failures_of(row: dict[str, Any]) -> list[str]:
    """The §3.2 violations of one serialized row."""
    out = []
    for check in row["inputs"]:
        if check["verdict"] in ("shape_differs", "rank_differs", "kind_differs",
                                "missing_in_phase0"):
            out.append(f"input `{check['name']}`: {check['verdict']}: {check['detail']}")
        out += [
            f"input `{check['name']}` axis {a['axis']}: above the baseline's "
            f"bound of {a['baseline']} ({a['expression']})"
            for a in check["axes"]
            if a["verdict"] == "looser"
        ]
    for match in row["dynamic_arrays"]:
        out += [
            f"{match['phase0']} axis {a['axis']}: above the baseline's bound "
            f"of {a['baseline']} ({a['expression']})"
            for a in match["axes"]
            if a["verdict"] == "looser"
        ]
    out += [
        f"return {check['path']}: {check['verdict']} ({check['detail']})"
        for check in row["returns"]
        if check["verdict"] == "looser"
    ]
    return out


def _verdict_table(summary: dict[str, Any]) -> str:
    keys = [
        ("input descriptors", "input_verdicts"),
        ("static arrays", "static_matches"),
        ("dynamic envelopes", "dynamic_matches"),
        ("dynamic axes", "dynamic_axis_verdicts"),
        ("returned values", "return_verdicts"),
    ]
    return table(
        ["relation", "verdicts"],
        [
            [
                label,
                ", ".join(
                    f"{count} {name}"
                    for name, count in sorted(summary[key].items(), key=lambda kv: -kv[1])
                )
                or "none",
            ]
            for label, key in keys
        ],
    )


def _tightening_table(rows: list[dict[str, Any]]) -> str:
    entries: list[Sequence[Any]] = []
    for row in rows:
        label = row["program"] + "::" + row["function"]
        for check in row["inputs"]:
            entries += [
                [label, f"input `{check['name']}`", a["axis"], a["baseline"],
                 a["phase0"], a["expression"]]
                for a in check["axes"]
                if a["verdict"] == "tighter"
            ]
        for match in row["dynamic_arrays"]:
            entries += [
                [label, match["phase0"], a["axis"], a["baseline"], a["phase0"],
                 a["expression"]]
                for a in match["axes"]
                if a["verdict"] == "tighter"
            ]
        entries += [
            [label, "the return payload", "—", check["detail"], check["bounds"],
             ", ".join(check["dims"])]
            for check in row["returns"]
            if check["verdict"] == "tighter"
        ]
    shown, note = capped(entries)
    return table(
        ["circuit", "array", "axis", "baseline", "phase 0", "expression"], shown
    ) + note


def _unmatched_table(rows: list[dict[str, Any]]) -> str:
    """Unmatched arrays on both sides; a grouped row counts for its group."""
    counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        for match in row["static_arrays"]:
            if match["verdict"] not in ("matched", "dynamic_backing"):
                counts[("static", match["verdict"])] += match.get("count", 1)
        for match in row["dynamic_arrays"]:
            if match["match"].startswith("unmatched"):
                counts[("dynamic", match["match"])] += match.get("count", 1)
    return table(
        ["relation", "side", "arrays"],
        [[kind, verdict, count] for (kind, verdict), count in counts.most_common()],
    )


def _check_table(rows: list[dict[str, Any]]) -> str:
    """Answers by verdict: distinct questions, and the checks they stand for.

    Phase 0 records one check per operation per unrolled iteration, and the
    term arena is hash-consed, so many are the same `(path, defcond)`
    question; the harness answers each distinct question once.
    """
    distinct: Counter[str] = Counter()
    total: Counter[str] = Counter()
    for row in rows:
        for result in row["checks"]:
            distinct[result["answer"]] += result["distinct"]
            total[result["answer"]] += result["occurrences"]
    return table(
        ["answer", "distinct questions", "checks"],
        [[answer, distinct[answer], count] for answer, count in total.most_common()],
    )


def _facts_table(rows: list[dict[str, Any]]) -> str:
    enforced: Counter[str] = Counter()
    checks = 0
    exact = 0
    for row in rows:
        facts = row.get("facts") or {}
        enforced.update(facts.get("baseline_enforced", {}))
        checks += facts.get("phase0_checks", 0)
        exact += len(facts.get("exact_shape_resolutions", []))
    return "\n".join(
        [
            table(
                ["baseline IR check", "statements"],
                [[name, count] for name, count in enforced.most_common()],
            ),
            "",
            table(
                ["phase 0", "count"],
                [["checks in Φ", checks],
                 ["exact symbolic shape resolutions the baseline made", exact]],
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------

#: The fields of `record.summary`, as the per-family table prints them.
RECORD_FIELDS = ("entries", "named", "merges", "input", "alias", "opaque",
                 "applied")


def section_record(rows: list[dict[str, Any]] | None) -> str:
    """$\\Delta$ per circuit: what it holds and what it costs.

    One entry per keyed value (`implementation.md` §5a), so the counts say
    how much of the record is a value with a source name, how much an
    intermediate, and how each value was produced. A circuit that broke a
    record invariant is a `invariant_violation` row of the coverage section,
    not of this one, so "invariants" here is a count of the circuits that
    passed.
    """
    if rows is None:
        return "_not yet produced: run `testing/phase0/test_coverage.py`._\n"
    accepted = [row for row in rows if row["status"] == "ok" and row.get("record")]
    if not accepted:
        return "_no accepted circuit carries a record summary._\n"
    families = sorted({row["family"] for row in accepted})
    per_family = [
        _record_summary(family, [r for r in accepted if r["family"] == family])
        for family in families
    ]
    per_family.append(_record_summary("**total**", accepted))
    largest = sorted(accepted, key=lambda r: -r["record"]["entries"])[:DETAIL_ROWS]
    violations = sum(1 for row in rows if row["status"] == "invariant_violation")
    return "\n".join(
        [
            f"The record of `implementation.md` §5a on the {len(accepted)} "
            "accepted circuits: entries by source name, by producer, and the "
            "record's size as JSON. Each cell is median / maximum.\n",
            table(
                ["family", "circuits", "entries", "named", "merges",
                 "input", "alias", "opaque", "applied", "MB"],
                per_family,
            ),
            f"\nThe {len(largest)} largest records by entry count.\n",
            table(
                ["circuit", "entries", "named", "merges", "alias", "applied", "MB"],
                [
                    [
                        f"{r['program']}::{r['function']}",
                        r["record"]["entries"], r["record"]["named"],
                        r["record"]["merges"], r["record"]["alias"],
                        r["record"]["applied"],
                        f"{r['record']['bytes'] / 1e6:.2f}",
                    ]
                    for r in largest
                ],
            ),
            f"\nRecord invariants (I1 to I5, `record.py`): "
            f"{len(accepted) - violations} of {len(accepted)} accepted "
            f"circuits hold them, {violations} violations.\n",
        ]
    )


def _record_summary(label: str, rows: list[dict[str, Any]]) -> list[Any]:
    def pair(field: str) -> str:
        values = [row["record"][field] for row in rows]
        return f"{int(median(values))} / {max(values)}"

    sizes = [row["record"]["bytes"] / 1e6 for row in rows]
    return [
        label, len(rows), *[pair(field) for field in RECORD_FIELDS],
        f"{median(sizes):.2f} / {max(sizes):.2f}",
    ]


# ---------------------------------------------------------------------------
# Size
# ---------------------------------------------------------------------------


def source_lines(path: Path) -> int:
    """Non-blank, non-comment lines of a kernel file."""
    if not path.exists():
        return 0
    return sum(
        1
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.strip() and not line.strip().startswith("#")
    )


def size_rows(
    coverage: list[dict[str, Any]] | None, corpus: "CorpusFacts"
) -> list[dict[str, Any]]:
    """One row per accepted circuit: program size beside $\\lvert\\Phi\\rvert$."""
    if coverage is None:
        return []
    out = []
    for row in coverage:
        if row["status"] != "ok":
            continue
        label = f"{row['program']}::{row['function']}"
        baseline = corpus.baselines.get(label)
        out.append(
            {
                "family": row["family"],
                "label": label,
                "sloc": corpus.source_lines.get(label, 0),
                "ir": baseline.ir_statements if baseline and baseline.ok else None,
                "variables": row["variables"],
                "constraints": row["constraints"],
                "terms": row["terms"],
            }
        )
    return out


def section_size(coverage: list[dict[str, Any]] | None, corpus: "CorpusFacts") -> str:
    rows = size_rows(coverage, corpus)
    if not rows:
        return "_not yet produced: run `testing/phase0/test_coverage.py`._\n"
    families = sorted({row["family"] for row in rows})
    per_family = [
        _size_summary(family, [r for r in rows if r["family"] == family])
        for family in families
    ]
    per_family.append(_size_summary("**total**", rows))
    largest = sorted(rows, key=lambda r: -r["terms"])[:DETAIL_ROWS]
    return "\n".join(
        [
            "$\\lvert\\Phi\\rvert$ against program size, per family; each cell is "
            "median / maximum over the accepted circuits.\n",
            table(
                ["family", "circuits", "source lines", "baseline IR stmts",
                 "variables", "constraints", "term nodes"],
                per_family,
            ),
            f"\nThe {len(largest)} largest records by term count.\n",
            table(
                ["circuit", "source lines", "baseline IR stmts", "|V|", "|C|", "terms"],
                [
                    [r["label"], r["sloc"], r["ir"] if r["ir"] is not None else "—",
                     r["variables"], r["constraints"], r["terms"]]
                    for r in largest
                ],
            ),
        ]
    )


def _size_summary(label: str, rows: list[dict[str, Any]]) -> list[Any]:
    def pair(key: str) -> str:
        values = [r[key] for r in rows if r[key] is not None]
        return f"{int(median(values))} / {max(values)}" if values else "—"

    return [
        label, len(rows), pair("sloc"), pair("ir"),
        pair("variables"), pair("constraints"), pair("terms"),
    ]


# ---------------------------------------------------------------------------
# Defects
# ---------------------------------------------------------------------------

#: The heading that opens `coverage.md` §9 and the one that closes its table.
_SECTION_9 = re.compile(r"^## 9\.", re.MULTILINE)


def coverage_defect_rows() -> list[list[str]]:
    """The rows of `coverage.md` §9's first table, its one source of truth."""
    if not COVERAGE_MD.exists():
        return []
    text = COVERAGE_MD.read_text(encoding="utf-8")
    match = _SECTION_9.search(text)
    if match is None:
        return []
    rows = []
    for line in text[match.end():].splitlines():
        if not line.startswith("|"):
            if rows:
                break
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 4 and not set(cells[0]) <= {"-", " "}:
            rows.append(cells)
    return rows[1:] if rows else []


def section_defects(
    coverage: list[dict[str, Any]] | None, corpus: "CorpusFacts"
) -> str:
    excluded = corpus.coverage_defects
    rejected = (
        [r for r in coverage if r["status"] != "ok"] if coverage is not None else []
    )
    baseline_failures = corpus.baseline_failures()
    return "\n".join(
        [
            "Baseline defects phase 0 does not replicate (`coverage.md` §9); "
            "programs that depend on one are outside the parity set.\n",
            table(
                ["baseline behaviour", "location", "phase 0 behaviour", "programs"],
                excluded,
            ),
            "\nParity circuits phase 0 rejects; every one is a §9 row, and the "
            "reason names what is missing.\n",
            histogram(Counter(r["reason"] for r in rejected)),
            "\nParity circuits the baseline itself no longer compiles or cannot "
            "compile within the harness budget — the evidence for the two stale "
            "record entries of `coverage.md` §9 and for the cost exclusions.\n",
            table(["circuit", "baseline status", "message"], baseline_failures),
        ]
    )


# ---------------------------------------------------------------------------
# The document
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusFacts:
    """Everything the report reads from outside the four result JSONs.

    Gathered once, in :func:`build`, so that every `section_*` is a pure
    function of data a test can hand it: the kernel sources, the cached
    baseline records and `coverage.md` §9's table.
    """

    #: Non-blank source lines per circuit label.
    source_lines: dict[str, int]
    #: The cached baseline record per circuit label, where one exists.
    baselines: dict[str, BaselineRecord]
    #: `coverage.md` §9's first table, as rows of cells.
    coverage_defects: list[list[str]]

    @staticmethod
    def read() -> "CorpusFacts":
        sources: dict[str, int] = {}
        baselines: dict[str, BaselineRecord] = {}
        for circuit in parity_set():
            sources[circuit.label] = source_lines(circuit.kernel_path)
            path = cache_path(circuit.program, circuit.function)
            if path.exists():
                baselines[circuit.label] = BaselineRecord.from_json(
                    json.loads(path.read_text(encoding="utf-8"))
                )
        return CorpusFacts(sources, baselines, coverage_defect_rows())

    def baseline_failures(self) -> list[list[str]]:
        """Every parity circuit whose baseline record is not `compiled`."""
        return [
            [label, record.status, (record.error_message or "")[:160]]
            for label, record in self.baselines.items()
            if not record.ok
        ]


def build() -> str:
    coverage = load(RESULTS / "coverage.json")
    adequacy = load(RESULTS / "adequacy.json")
    hook = load(RESULTS / "adequacy_hook.json")
    consistency = load(RESULTS / "consistency.json")
    corpus = CorpusFacts.read()
    return "\n".join(
        [
            "# Phase 0 acceptance report",
            "",
            "Generated by `testing/phase0/report.py` from the three harness "
            "result files in this directory. The sections are those of "
            "`revision/phase0/acceptance.md` §5.",
            "",
            "## 1. Coverage",
            "",
            section_coverage(coverage),
            "## 2. Adequacy",
            "",
            section_adequacy(adequacy, hook),
            "## 3. Consistency",
            "",
            section_consistency(consistency),
            "## 4. Record",
            "",
            section_record(coverage),
            "## 5. Size",
            "",
            section_size(coverage, corpus),
            "## 6. Defects",
            "",
            section_defects(coverage, corpus),
        ]
    )


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text(build(), encoding="utf-8")
    print(f"wrote {REPORT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
