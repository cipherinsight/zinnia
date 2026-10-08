"""The classification table of the scaled points, in markdown.

Reads `DATA/points.json` (written by `generate`) and, when given, validation
reports (`validate` outputs, `baselines/zkvm_common/check_outputs.py
--report` outputs, labelled with where they ran).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from benchmarking.scaling_npb.generate import FIXED, SCALED, UNCHANGED


def load_points(data: Path) -> dict:
    return json.loads((data / "points.json").read_text())


def base_case(case: str) -> str:
    return case.rsplit("_cap", 1)[0]


def cap_of(case: str) -> int:
    return int(case.rsplit("_cap", 1)[1])


def summary_rows(points: list[dict]) -> list[str]:
    caps = sorted({p["cap"] for p in points})
    lines = ["| CAP | corpus | scaled | unchanged | fixed size |", "|---|---|---|---|---|"]
    for cap in caps:
        for group in ("npbench", "pythran", "all"):
            mine = [p for p in points if p["cap"] == cap and group in (p["group"], "all")]
            count = Counter(p["status"] for p in mine)
            lines.append(f"| {cap} | {group} | {count[SCALED]} | {count[UNCHANGED]} | {count[FIXED]} |")
    return lines


def cell(point: dict) -> str:
    if point["status"] == SCALED:
        how = f"d={point['divisor']}" if point.get("divisor") else f"t={point.get('largest')}"
        return f"{point['input_elements']:,} el. ({how})"
    if point["status"] == UNCHANGED:
        return f"= CAP {point['same_as']}"
    return "fixed"


def entry_rows(document: dict) -> list[str]:
    points = document["points"]
    caps = sorted({p["cap"] for p in points})
    by_case: dict[str, dict[int, dict]] = defaultdict(dict)
    for p in points:
        by_case[base_case(p["case"])][p["cap"]] = p
    lines = ["| entry | CAP 16 (perf set) | " + " | ".join(f"CAP {c}" for c in caps) + " | knobs at the largest cap |",
             "|---|---|" + "---|" * len(caps) + "---|"]
    for case in sorted(by_case):
        relation = document["reference_cap"].get(case, {})
        row = [case, relation.get("relation", "")]
        row += [cell(by_case[case][c]) for c in caps]
        last = by_case[case][caps[-1]]
        row.append(last["reason"] if last["status"] == FIXED or last.get("reason") else str(last.get("knobs") or ""))
        lines.append("| " + " | ".join(row) + " |")
    return lines


def validation_rows(reports: list[Path]) -> list[str]:
    lines = []
    for path in reports:
        if path.suffix == ".jsonl":                                     # validate.py
            records = [json.loads(line) for line in path.read_text().splitlines() if line]
            by_system: dict[tuple, Counter] = defaultdict(Counter)
            for record in records:
                by_system[(record["cap"], record["system"], record["host"])][record["outcome"]] += 1
            for (cap, system, where), counts in sorted(by_system.items()):
                lines.append(f"| {cap} | {system}, mock vs NumPy | {where} | "
                             + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) + " |")
        else:                                                           # check_outputs.py --report
            document = json.loads(path.read_text())
            by_system: dict[str, Counter] = defaultdict(Counter)
            for record in document:
                verdict = "skipped" if record["skipped"] else "ok" if record["ok"] else "FAIL"
                by_system[record["system"]][verdict] += 1
            cap = path.stem.rsplit("cap", 1)[-1]
            for system, counts in sorted(by_system.items()):
                lines.append(f"| {cap} | {system} execute vs NumPy | local | "
                             + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) + " |")
    if lines:
        lines = ["| CAP | check | where | outcomes |", "|---|---|---|---|"] + lines
    return lines


def render(data: Path, validation: list[Path]) -> str:
    document = load_points(data)
    parts = ["## Classification per cap", "", *summary_rows(document["points"]), "",
             "## Per entry", "", "Cells: input elements of input A and, for a scaled point, the divisor d (NPBench) or the "
             "largest knob's value t (Pythran, proportional formula); "
             "`= CAP c` for a point with the sizes of an earlier one; `fixed` for one the rule cannot scale. "
             "CAP 16 column: how the performance set's sizes relate to the rule at CAP 16.", "",
             *entry_rows(document)]
    rows = validation_rows(validation)
    if rows:
        parts += ["", "## Validation", "", *rows]
    return "\n".join(parts) + "\n"
