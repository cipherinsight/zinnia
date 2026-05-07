from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReportRow:
    case_id: str
    operator: str
    kind: str
    spelling: str
    mode: str
    dtype: str
    rank: int | None
    shape_pattern: str
    expected_status: str
    actual_status: str
    failure_class: str | None
    manifest_reason: str | None
    replaces: str | None


_ROWS: list[ReportRow] = []


def record(row: ReportRow) -> None:
    _ROWS.append(row)


def rows() -> list[ReportRow]:
    return list(_ROWS)


def write_report(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter(row.actual_status for row in _ROWS)
    expected_counts = Counter(row.expected_status for row in _ROWS)
    payload: dict[str, Any] = {
        "summary": {
            "passing_semantic_cells": counts["pass"],
            "known_gaps": expected_counts["xfail"],
            "intentional_rejections": expected_counts["reject"],
            "unexpected_failures": counts["unexpected_failure"],
            "unclear_contract_cells": expected_counts["unclear"],
            "total_cells": len(_ROWS),
        },
        "by_expected_status": dict(expected_counts),
        "by_actual_status": dict(counts),
        "cases": [asdict(row) for row in _ROWS],
    }
    import json

    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
