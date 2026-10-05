"""Graph size against program size (`revision/phase1/acceptance.md` §2.2;
`revision/README.md` §11, row 1: the graph is polynomial in Ω).

Per accepted circuit: |Δ| (entries of the record), |Φ| (constraints),
checks, requirement nodes, implementation nodes still in G, pruned nodes,
rows of the provider table, terms phase 1 built; and the ratio of nodes
(requirements plus implementations) to |Δ|, whose spread over the corpus is
what the hypothesis is read from.

```text
.venv/bin/python -m testing.phase1.sizes
```
"""

from __future__ import annotations

from collections import Counter
from statistics import median
from typing import Any, Iterable

from testing.phase1.cache import CACHE_DIR, Entry, entries


def rows(entry: Entry) -> dict[str, Any] | None:
    """One row per accepted circuit; `None` for the rest."""
    if not entry.accepted or entry.phase0 is None or entry.phase1 is None:
        return None
    record, result = entry.phase0, entry.phase1
    stats = result["stats"]
    delta = len(record["descriptors"])
    requirements = len(result["requirements"])
    implementations = int(stats["implementations"])
    nodes = requirements + implementations
    rules = Counter(n["rule"] for n in result["implementations"] if not n["dropped"])
    return {
        "top_rules": [f"{rule} {count}" for rule, count in rules.most_common(3)],
        "label": entry.label,
        "family": entry.family,
        "delta": delta,
        "phi": len(record["constraints"]),
        "variables": len(record["variables"]),
        "checks": sum(1 for c in record["constraints"] if c["kind"] == "check"),
        "phase0_terms": len(record["terms"]["nodes"]),
        "requirements": requirements,
        "implementations": implementations,
        "all_implementations": len(result["implementations"]),
        "pruned": int(stats["pruned"]),
        "roots": len(result["roots"]),
        "rows": int(stats["rows"]),
        "terms_built": int(stats["terms_built"]),
        "answers": len(result["answers"]),
        "nodes": nodes,
        "ratio": nodes / delta if delta else 0.0,
        "seconds_phase1": max(
            0.0,
            float(entry.seconds.get("candidates", 0.0)) - float(entry.seconds.get("analyze", 0.0)),
        ),
    }


def _spread(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "median": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": min(values),
        "median": float(median(values)),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def summary(table: list[dict[str, Any]]) -> dict[str, Any]:
    """Totals, the spread of the ratio, and the largest graphs."""
    ratios = [r["ratio"] for r in table]
    by_nodes = sorted(table, key=lambda r: -r["nodes"])
    return {
        "circuits": len(table),
        "delta": sum(r["delta"] for r in table),
        "requirements": sum(r["requirements"] for r in table),
        "implementations": sum(r["implementations"] for r in table),
        "pruned": sum(r["pruned"] for r in table),
        "terms_built": sum(r["terms_built"] for r in table),
        "ratio": _spread(ratios),
        "ratio_over_10": [r["label"] for r in table if r["ratio"] > 10],
        "largest": [r["label"] for r in by_nodes[:5]],
        "highest_ratio": [(r["label"], r["ratio"], r["top_rules"])
                          for r in sorted(table, key=lambda r: -r["ratio"])[:3]],
        "lowest_ratio": [(r["label"], r["ratio"], r["top_rules"])
                         for r in sorted(table, key=lambda r: r["ratio"])[:3]],
        "median_row": {
            key: float(median(r[key] for r in table)) if table else 0.0
            for key in ("delta", "phi", "checks", "requirements", "implementations",
                        "pruned", "rows", "terms_built")
        },
    }


def table(source: Iterable[Entry]) -> list[dict[str, Any]]:
    return [row for row in (rows(entry) for entry in source) if row is not None]


def main() -> int:
    result = table(entries(CACHE_DIR))
    header = (f"{'circuit':<46}{'|Δ|':>7}{'|Φ|':>6}{'chk':>5}{'req':>6}{'impl':>6}"
              f"{'pruned':>7}{'rows':>6}{'terms':>7}{'nodes/|Δ|':>10}")
    print(header)
    print("-" * len(header))
    for row in sorted(result, key=lambda r: -r["nodes"]):
        print(f"{row['label'][:45]:<46}{row['delta']:>7}{row['phi']:>6}{row['checks']:>5}"
              f"{row['requirements']:>6}{row['implementations']:>6}{row['pruned']:>7}"
              f"{row['rows']:>6}{row['terms_built']:>7}{row['ratio']:>10.2f}")
    totals = summary(result)
    print("-" * len(header))
    print(f"{totals['circuits']} circuits, {totals['requirements']} requirement and "
          f"{totals['implementations']} implementation nodes over |Δ| = {totals['delta']}; "
          f"nodes/|Δ| min {totals['ratio']['min']:.2f}, median {totals['ratio']['median']:.2f}, "
          f"max {totals['ratio']['max']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
