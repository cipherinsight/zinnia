#!/usr/bin/env python3
"""Compare two `measure.py` outputs (a configuration against the compiler, `final`).

    python benchmarking/ablations/compare.py measure_final.json measure_abl_joint.json [--markdown]

Lists every case and target whose circuit differs (IR digest), with its cost
(halo2-base cells and `k`, Noir gates when measured) in both, the ratio, and the
families whose chosen constructions differ; then a summary per case set and target:
cases changed, costlier, cheaper, the geometric mean of the ratio over every measured
case, the fallback groups of the configuration, and the groups lowered through the
backend's memory in each.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def chosen_diff(a: dict, b: dict) -> list[str]:
    ca, cb = a.get("layer", {}).get("chosen", {}), b.get("layer", {}).get("chosen", {})
    return [f"{family}: {ca.get(family, {})} -> {cb.get(family, {})}"
            for family in sorted(set(ca) | set(cb)) if ca.get(family) != cb.get(family)]


def geomean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base", type=Path)
    ap.add_argument("other", type=Path)
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()
    base, other = (json.loads(p.read_text()) for p in (args.base, args.other))
    if args.markdown:
        print("| set | case | target | base cost (k) | other cost (k) | ratio | families changed |")
        print("|---|---|---|---|---|---|---|")
    summary = []
    for set_name in sorted(k for k in base if k != "config"):
        for target in common.TARGETS:
            ratios, changed, costlier, cheaper = [], 0, 0, 0
            fallback = native_base = native_other = 0
            for case in sorted(base[set_name]):
                a, b = base[set_name][case].get(target), other.get(set_name, {}).get(case, {}).get(target)
                if a is None or b is None:
                    continue
                fallback += b.get("layer", {}).get("fallback", 0)
                native_base += a.get("layer", {}).get("native_memory", 0)
                native_other += b.get("layer", {}).get("native_memory", 0)
                ca, cb = common.cost_of(a, target), common.cost_of(b, target)
                if ca and cb:
                    ratios.append(cb / ca)
                    costlier += cb > ca
                    cheaper += cb < ca
                if a.get("ir") == b.get("ir") and "error" not in b:
                    continue
                changed += 1
                ratio = f"{cb / ca:.3f}" if ca and cb else "-"
                ka, kb = a.get("k"), b.get("k")
                cost_a = f"{ca}" + (f" ({ka})" if ka else "")
                cost_b = f"{cb}" + (f" ({kb})" if kb else "") if "error" not in b else b["error"][:80]
                families = "; ".join(chosen_diff(a, b))
                if args.markdown:
                    print(f"| {set_name} | {case.removeprefix('ds1000_case')} | {target} | {cost_a} | "
                          f"{cost_b} | {ratio} | {families} |")
                else:
                    print(f"{set_name} {case} {target}: {cost_a} -> {cost_b} ratio {ratio}  {families}")
            g = geomean(ratios)
            summary.append(f"{set_name} {target}: {changed} circuits differ, {costlier} costlier, "
                           f"{cheaper} cheaper of {len(ratios)} measured; geomean ratio "
                           f"{f'{g:.4f}' if g else '-'}; fallback groups {fallback}; "
                           f"native ROM/RAM groups {native_base} -> {native_other}")
    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
