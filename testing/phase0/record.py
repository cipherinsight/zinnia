"""The record invariants of `revision/phase0/implementation.md` §5a, over the
exported JSON, and the summary of $\\Delta$ the report prints.

The same five checks as `src/phase0/tests/invariants.rs`, on the other side of
the export: the coverage worker runs them on every accepted circuit, so a
producer, an operand or a subject that a rule forgets is a failure row of the
parity run rather than something phase 1 discovers.

- I1 every entry has a producer; an `applied` producer of a scalar or an array
  names at least one rule; `opaque` stands only for a value with no descriptor
  content (a dtype or module name, or `None`).
- I2 every operand and every alias resolves to an entry recorded *earlier* in
  Δ, so the traversal order is a topological order of the source graph.
- I3 every check constraint has a subject that resolves to an entry.
- I4 keys are unique.
- I5 the check watermarks do not fall along Δ and none exceeds the number of
  check constraints: Φ only grows, and every watermark is the count it held
  when the entry was recorded.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

#: The descriptor kinds that carry no content of their own (§3.3).
OPAQUE_KINDS = ("class", "none")


def summary(record: dict[str, Any]) -> dict[str, int]:
    """What one circuit's $\\Delta$ contributes to `results/REPORT.md`.

    `bytes` is the record as `json.dumps` writes it, which is how every
    measurement of the record's size has been taken.
    """
    entries = record["descriptors"]
    producers = Counter(entry["producer"]["kind"] for entry in entries)
    return {
        "entries": len(entries),
        "named": sum(1 for entry in entries if entry.get("name")),
        "merges": sum(1 for entry in entries if _is_merge(entry["producer"])),
        "input": producers["input"],
        "alias": producers["alias"],
        "opaque": producers["opaque"],
        "applied": producers["applied"],
        "bytes": len(json.dumps(record)),
    }


def _is_merge(producer: dict[str, Any]) -> bool:
    """Whether a producer applied one of §6.1's merge rules."""
    return any(
        rule.startswith("merge-") for rule in producer.get("rules") or []
    )


def violations(record: dict[str, Any], limit: int = 20) -> list[str]:
    """Every invariant an accepted record breaks, one line each."""
    problems: list[str] = []
    seen: set[str] = set()
    checks = sum(1 for c in record["constraints"] if c["kind"] == "check")
    watermark = 0
    for entry in record["descriptors"]:
        key = entry["key"]["key"]
        problems.extend(_producer_problems(key, entry))
        # Checked before the key joins `seen`, so a value that names itself is
        # a violation and not a cycle of length one.
        for other in _inputs(entry["producer"]):
            if other not in seen:
                problems.append(
                    f"I2: the producer of {key} names {other}, "
                    "which Δ does not record before it"
                )
        if key in seen:
            problems.append(f"I4: two values share the key {key}")
        seen.add(key)
        problems.extend(_watermark_problems(key, entry, watermark, checks))
        watermark = entry["checks"]
        if len(problems) >= limit:
            return problems[:limit]
    problems.extend(_subject_problems(record, seen))
    return problems[:limit]


def _producer_problems(key: str, entry: dict[str, Any]) -> list[str]:
    """I1 for one entry."""
    producer = entry.get("producer")
    if producer is None:
        return [f"I1: {key} has no producer"]
    kind = entry["desc"]["kind"]
    opaque = kind in OPAQUE_KINDS
    if producer["kind"] == "opaque" and not opaque:
        return [f"I1: {key} is an entry of kind {kind} and may not be opaque"]
    if producer["kind"] == "applied" and not (producer.get("rules") or []):
        # A composite applies no rule of its own: it is a list of descriptors
        # (§3.3), and the rules ran on its elements.
        if kind != "composite" and not opaque:
            return [f"I1: {key} names no rule"]
    return []


def _watermark_problems(
    key: str, entry: dict[str, Any], previous: int, checks: int
) -> list[str]:
    """I5 for one entry: its watermark is the number of checks Φ held when it
    was recorded, so it lies between the previous entry's and the number of
    checks Φ ends up holding.
    """
    watermark = entry["checks"]
    if watermark < previous:
        return [f"I5: the check watermark of {key} is {watermark}, "
                f"below the {previous} of the entry before it"]
    if watermark > checks:
        return [f"I5: the check watermark of {key} is {watermark}, "
                f"above the number of checks in Φ ({checks})"]
    return []


def _subject_problems(record: dict[str, Any], seen: set[str]) -> list[str]:
    """I3 over the checks."""
    problems: list[str] = []
    for constraint in record["constraints"]:
        if constraint["kind"] != "check":
            continue
        subject = constraint["subject"]
        if subject not in seen:
            problems.append(
                f"I3: the check from {constraint['provenance']} points at "
                f"{subject}, which Δ does not record"
            )
    return problems


def _inputs(producer: dict[str, Any]) -> list[str]:
    """The keys a producer reads: its operands, or the value it aliases.

    A reference is the canonical key string, not the key object the entry
    itself carries (`export.rs`'s schema).
    """
    if producer["kind"] == "alias":
        return [producer["alias"]]
    return list(producer.get("operands") or [])
