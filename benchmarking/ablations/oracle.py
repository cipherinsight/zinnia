#!/usr/bin/env python3
"""Oracle comparison: is the layer's selection ever beaten by a fixed construction?

For every `perf` case of the case sets and each target:

1. compile with the compiler's selection (no switch) and read, from the layer report,
   the families of operators the program has and the constructions applicable to
   each family's groups;
2. for every family with two or more applicable constructions, compile once per
   construction with that construction forced (`ZINNIA_FORCE_CONSTRUCTION=<name>`):
   every group of the family it applies to uses it, the other families keep the
   layer's choice (one family at a time);
3. report, per (case, target, family), the cost of the compiler's choice against the
   best and the worst forced construction, and whether a forced one is cheaper
   (`beaten`).

Costs are the measured ones: halo2-base advice + lookup cells at the `k` that
`estimate_params` picks (a forced construction can change `k`, and with it the lookup
width), and Noir `bb gates`. On halo2-base the verdict is also given in `poly_cost`
(committed polynomials x 2^k, `poly_*` keys), the size model that counts the columns of
fixed overheads (a verifier challenge, lookup tables) the cell count does not see. A forced construction that applies to only some groups of
its family is reported with `applied` < `groups` (the rest keep the layer's choice).

    python benchmarking/ablations/oracle.py --out oracle.json [--case-sets ds1000,npbench_pythran]
        [--cases ...] [--targets halo2,noir] [--jobs 8] [--timeout 1200]
        [--wrap "python benchmarking/ablations/capped.py 8G 1100"]
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def alternatives(layer: dict) -> dict[str, list[str]]:
    """Per family, the constructions applicable to some of its groups (helpers left
    out), in order of first appearance; only families with two or more."""
    by_family: dict[str, list[str]] = {}
    for group in layer.get("groups", []):
        names = by_family.setdefault(group["family"], [])
        for name in group["applicable"]:
            if name not in names and name not in common.HELPERS:
                names.append(name)
    return {family: names for family, names in by_family.items() if len(names) >= 2}


def applied(layer: dict, family: str, name: str) -> tuple[int, int]:
    """(groups of `family` whose lanes come from `name`, groups of `family`)."""
    groups = [g for g in layer.get("groups", []) if g["family"] == family]
    return sum(name in g["chosen"] for g in groups), len(groups)


def verdict(default_cost: int | None, forced: dict[str, dict], key: str = "cost", prefix: str = "") -> dict:
    """The best and worst forced construction by `key` against the compiler's choice."""
    costs = {name: r[key] for name, r in forced.items() if r.get(key) is not None}
    if default_cost is None or not costs:
        return {f"{prefix}best": None, f"{prefix}worst": None, f"{prefix}beaten": None}
    best = min(costs, key=lambda n: (costs[n], n))
    worst = max(costs, key=lambda n: (costs[n], n))
    return {f"{prefix}best": [best, costs[best]], f"{prefix}worst": [worst, costs[worst]],
            f"{prefix}beaten": costs[best] < default_cost,
            f"{prefix}best_over_default": round(costs[best] / default_cost, 4),
            f"{prefix}worst_over_default": round(costs[worst] / default_cost, 4)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case-sets", default=",".join(common.CASE_SETS))
    ap.add_argument("--cases", help="comma-separated case names (default: every perf case)")
    ap.add_argument("--targets", default=",".join(common.TARGETS))
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=1200)
    ap.add_argument("--wrap", default="", help="command prefix of each child (a memory cap)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    wanted = set(args.cases.split(",")) if args.cases else None
    cases = []
    for set_name in args.case_sets.split(","):
        entries = common.perf_entries(common.CASE_SETS[set_name])
        for case in sorted(entries):
            if wanted is None or case in wanted or case.removeprefix("ds1000_case") in wanted:
                for target in args.targets.split(","):
                    cases.append((set_name, case, entries[case], target))
    wrap = shlex.split(args.wrap)

    def compile_one(entry: dict, target: str, env: dict[str, str]) -> dict:
        return common.run_child(entry, target, env, target == "noir", args.timeout, wrap)

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        defaults = list(pool.map(lambda c: compile_one(c[2], c[3], {}), cases))
        forced_jobs = [(i, family, name) for i, record in enumerate(defaults)
                       for family, names in alternatives(record.get("layer", {})).items()
                       for name in names]
        forced_records = list(pool.map(
            lambda job: compile_one(cases[job[0]][2], cases[job[0]][3],
                                    {"ZINNIA_FORCE_CONSTRUCTION": job[2]}), forced_jobs))

    results: dict = {}
    for i, (set_name, case, _, target) in enumerate(cases):
        default = defaults[i]
        row = {"default": {"cost": common.cost_of(default, target),
                           "poly_cost": common.cost_of(default, target, "poly_cost"),
                           "ir": default.get("ir"),
                           "k": default.get("k"), "error": default.get("error"),
                           "chosen": default.get("layer", {}).get("chosen")},
               "families": {}}
        results.setdefault(set_name, {}).setdefault(case, {})[target] = row
    for (i, family, name), record in zip(forced_jobs, forced_records):
        set_name, case, _, target = cases[i]
        row = results[set_name][case][target]
        uses, groups = applied(record.get("layer", {}), family, name)
        row["families"].setdefault(family, {"alternatives": {}})["alternatives"][name] = {
            "cost": common.cost_of(record, target), "poly_cost": common.cost_of(record, target, "poly_cost"),
            "k": record.get("k"), "ir": record.get("ir"),
            "applied": uses, "groups": groups, "error": record.get("error")}
    beaten = []
    for set_name, per_case in results.items():
        for case, per_target in per_case.items():
            for target, row in per_target.items():
                for family, entry in row["families"].items():
                    entry.update(verdict(row["default"]["cost"], entry["alternatives"]))
                    if target == "halo2":
                        entry.update(verdict(row["default"]["poly_cost"], entry["alternatives"],
                                             "poly_cost", "poly_"))
                    if entry["beaten"]:
                        poly = f" (poly cost: {'also beaten' if entry.get('poly_beaten') else 'not beaten'})" \
                            if target == "halo2" else ""
                        beaten.append((case, target, family, entry["best"], row["default"]["cost"], poly))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1, sort_keys=True) + "\n")
    compared = sum(len(r["families"]) for per in results.values() for t in per.values() for r in t.values())
    for case, target, family, best, cost, poly in beaten:
        print(f"BEATEN {case} {target} {family}: {best[0]} {best[1]} < selected {cost}{poly}")
    print(f"{len(beaten)} of {compared} (case, target, family) comparisons have a cheaper forced "
          f"construction; {len(cases)} case-targets, {len(forced_jobs)} forced compiles")
    return 0


if __name__ == "__main__":
    sys.exit(main())
