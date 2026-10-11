#!/usr/bin/env python3
"""Zinnia / baseline timing ratios per measurement session, and pooled over sessions.

A session is a set of epoch directories measured together (e.g. the 8 shards of `paired-r1`, or the
`paired-zkvm-z-*` Zinnia shards plus the `paired-zkvm-*` zkVM shards that ran on the same CPUs right
after them). Within a session each (system, case) has one record; its samples give a per-case median.

Per session: the per-case ratio is median(Zinnia samples) / median(baseline samples), and the summary
is the geometric mean over cases with the harness's percentile bootstrap (resampling cases).
Pooled: the per-case ratio is the geometric mean of that case's per-session ratios (each ratio is
paired within its session, so a slow session cancels out), with the same bootstrap over cases.

Cases: those `ok` for both systems in every session, split as the harness does into copy-only
(case set flag `data_movement_only`) and the rest (the headline).

Usage (from a zinnia worktree, for the harness modules):
  .venv/bin/python pooled.py --case-set FILE --pairs zinnia_halo2:halo2_hw,zinnia_noir:noir_hw \
      [--clocks prove_core_s,end_to_end_s] [--json OUT] --session NAME=DIR[,DIR...] [--session ...]
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())
from benchmarking.harness.analysis import aggregate, stats  # noqa: E402


def session_medians(dirs: list[Path], systems: set[str], clock: str) -> dict[str, dict[str, float]]:
    data = aggregate.load_shards(dirs)
    out: dict[str, dict[str, float]] = {}
    for system in systems:
        for case, record in data.records.get(system, {}).items():
            values = aggregate.clock_values(record, clock)
            if record["status"] == aggregate.OK and values:
                out.setdefault(system, {})[case] = stats.median(values)
    return out


def summarize(ratios: dict[str, float]) -> dict:
    values = list(ratios.values())
    if not values:
        return {"n": 0}
    low, high = stats.bootstrap_geomean_interval(values)
    return {"n": len(values), "geomean": stats.geometric_mean(values), "ci95": [low, high],
            "faster": sum(v < 1 for v in values), "slower": sum(v > 1 for v in values)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-set", type=Path, required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--clocks", default="prove_core_s,end_to_end_s")
    ap.add_argument("--session", action="append", required=True)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    raw = json.loads(args.case_set.read_text())
    copy_only = {e["case"] for e in (raw["cases"] if isinstance(raw, dict) else raw)
                 if e.get("data_movement_only")}
    pairs = [tuple(p.split(":")) for p in args.pairs.split(",")]
    systems = {s for p in pairs for s in p}
    sessions = []
    for spec in args.session:
        name, dirs = spec.split("=", 1)
        sessions.append((name, [Path(d) for d in dirs.split(",")]))
    report = {}
    for clock in args.clocks.split(","):
        medians = {name: session_medians(dirs, systems, clock) for name, dirs in sessions}
        for zinnia, baseline in pairs:
            common = set.intersection(*(set(m.get(zinnia, {})) & set(m.get(baseline, {}))
                                        for m in medians.values()))
            for subset, cases in (("headline", sorted(common - copy_only)),
                                  ("copy_only", sorted(common & copy_only))):
                per_session = {name: {c: m[zinnia][c] / m[baseline][c] for c in cases}
                               for name, m in medians.items()}
                pooled = {c: math.exp(sum(math.log(r[c]) for r in per_session.values()) / len(per_session))
                          for c in cases}
                key = f"{zinnia}/{baseline}/{clock}/{subset}"
                report[key] = {"sessions": {n: summarize(r) for n, r in per_session.items()},
                               "pooled": summarize(pooled), "per_case_pooled": pooled}
                line = "  ".join(f"{n} {s['geomean']:.3f} [{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}]"
                                 for n, s in report[key]["sessions"].items() if s["n"])
                p = report[key]["pooled"]
                if p["n"]:
                    print(f"{key} n={p['n']}: {line}  | pooled {p['geomean']:.3f} "
                          f"[{p['ci95'][0]:.3f}, {p['ci95'][1]:.3f}] faster/slower {p['faster']}/{p['slower']}")
    if args.json:
        args.json.write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
