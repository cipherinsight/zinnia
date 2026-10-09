"""Compare validated handwritten circuit sizes with a frozen Zinnia count file.

Run from benchmarking/baselines:
  python -m hw_check.summarize --checks check*.json --zinnia measure_final.json \
    --reference-root /path/to/frozen/checkout --out report.json

Every perf entry stays in the report, including missing or failed checks. Counts
are advice+lookup cells at each circuit's own sizing rule, or UltraHonk gates.
Fixture validation does not establish equivalence over the entire input domain.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from .entries import BENCH, npb_entries


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows: list[dict], target: str) -> dict:
    pairs = [r[target] for r in rows if r[target].get("ratio") is not None]
    ratios = [p["ratio"] for p in pairs]
    return {
        "entries": len(rows), "compared": len(pairs),
        "geomean_zinnia_over_handwritten": math.exp(sum(map(math.log, ratios)) / len(ratios)) if ratios else None,
        "zinnia_smaller": sum(r < 1 for r in ratios),
        "equal": sum(r == 1 for r in ratios),
        "zinnia_larger": sum(r > 1 for r in ratios),
    }


def make_report(checks: dict, measured: dict, reference_root: Path) -> dict:
    entries = {c: e for c, e in npb_entries().items() if e.status == "perf"}
    raw_cases = json.loads((BENCH / "datasets/case_sets/npbench_pythran_perf.json").read_text())["cases"]
    metadata = {c["case"]: c for c in raw_cases}
    rows = {}
    for case, entry in sorted(entries.items()):
        sources = {}
        for path in (entry.kernel, entry.fixture):
            reference = reference_root / path.relative_to(BENCH.parent)
            if not reference.is_file() or digest(path) != digest(reference):
                raise ValueError(f"frozen-reference mismatch: {path}")
            sources[path.name] = digest(path)
        for path in (entry.target_rs, entry.main_nr):
            if path.exists():
                sources[path.name] = digest(path)
        row = {"corpus": "NPBench" if case.startswith("npbench_") else "Pythran",
               "data_movement_only": bool(metadata[case].get("data_movement_only")),
               "sha256": sources}
        for target, field in (("halo2", "cells"), ("noir", "gates")):
            hw = checks.get(case, {}).get(target, {})
            z = measured.get(case, {}).get(target, {})
            counts = hw.get("cells") or {}
            size = counts.get("advice", 0) + counts.get("lookup", 0) if target == "halo2" and counts else hw.get("gates")
            result = {"validation": hw.get("verdict", "not_run"), "handwritten": size,
                      "zinnia": z.get(field), "ratio": None}
            if target == "halo2":
                result.update(handwritten_k=hw.get("k"), zinnia_k=z.get("k"))
            if hw.get("problems"):
                result["problems"] = hw["problems"]
            if checks.get(case, {}).get("error"):
                result["check_error"] = checks[case]["error"]
            if z.get("error"):
                result["zinnia_error"] = z["error"]
            if hw.get("verdict") == "pass" and not z.get("error") and size and z.get(field):
                result["ratio"] = z[field] / size
            row[target] = result
        rows[case] = row
    groups = {}
    for corpus in ("all", "NPBench", "Pythran"):
        selected = [r for r in rows.values() if corpus == "all" or r["corpus"] == corpus]
        groups[corpus] = {
            kind: {t: summarize(rs, t) for t in ("halo2", "noir")}
            for kind, rs in (("all_entries", selected),
                             ("excluding_data_movement", [r for r in selected if not r["data_movement_only"]]))}
    return {"metric": {"halo2": "advice + lookup cells, own sizing rule", "noir": "UltraHonk gates"},
            "scope": "fixture-validated circuit sizes; see baseline notes for full-domain limitations",
            "summary": groups, "cases": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checks", type=Path, nargs="+", required=True)
    ap.add_argument("--zinnia", type=Path, required=True)
    ap.add_argument("--reference-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    checks = {}
    for path in args.checks:
        for case, result in json.loads(path.read_text()).items():
            if case in checks:
                raise ValueError(f"duplicate check record {case}; provide one final validation per entry")
            checks[case] = result
    frozen = json.loads(args.zinnia.read_text())
    report = make_report(checks, frozen["npbench_pythran"], args.reference_root.resolve())
    report["provenance"] = {"zinnia": {"path": str(args.zinnia.resolve()), "sha256": digest(args.zinnia)},
                            "checks": [{"path": str(p.resolve()), "sha256": digest(p)} for p in args.checks]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# NPBench/Pythran circuit sizes", "", report["scope"] + ".", "",
             "Ratio = Zinnia / handwritten; below 1 means Zinnia is smaller.", "",
             "| Corpus | Entries | Target | Compared | Geomean ratio | Smaller / equal / larger |",
             "|---|---|---|---:|---:|---:|"]
    for corpus, kinds in report["summary"].items():
        for kind, targets in kinds.items():
            for target, summary in targets.items():
                gm = summary["geomean_zinnia_over_handwritten"]
                value = f"{gm:.4f}" if gm is not None else "—"
                lines.append(f"| {corpus} | {kind} | {target} | {summary['compared']}/{summary['entries']} | {value} | "
                             f"{summary['zinnia_smaller']} / {summary['equal']} / {summary['zinnia_larger']} |")
    lines += ["", "All per-entry counts, failures, sizing parameters and source hashes are in the companion JSON.",
              "No timing measurements are included.", ""]
    args.out.with_suffix(".md").write_text("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
