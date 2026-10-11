#!/usr/bin/env python3
"""Ablation tables from a run.py output directory.

    python analyze.py OUT [--top 8] [--metric size|poly_cost] [--json summary.json]

Per variant and backend, against `final` (the compiler as evaluated):
  * geometric mean of variant size / final size over the 214 programs and over the 172 that
    RQ2 compares (copy-only and constant-output programs dropped), each over the programs
    where both compiles succeeded (n is printed);
  * how many programs change (ratio != 1), grow, shrink, and how many change only the IR;
  * the largest increases, with program names;
  * failures: programs the variant fails on although final succeeds (and the reverse).
Sizes: halo2-base cells (advice + lookup of halo2_base_cell_stats, at the circuit's own k;
advice includes second-phase, table and RAM-trace advice) and Noir bb gates. A final size of 0 (constant-output halo2 circuits) gives
no ratio; such programs count as changed only if the variant's size is not 0 either.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def gmean(values: list[float]) -> float:
    return math.exp(sum(map(math.log, values)) / len(values)) if values else float("nan")


def short(case: str) -> str:
    return case.removeprefix("ds1000_case").split("__")[0]


def load(out: Path) -> tuple[list[dict], dict]:
    programs = json.loads((out / "programs.json").read_text())["programs"]
    results: dict = {}
    for path in (out / "results").glob("*/*/*.json"):
        variant, backend = path.parent.parent.name, path.parent.name
        results.setdefault(variant, {}).setdefault(backend, {})[path.stem] = json.loads(path.read_text())
    return programs, results


def compare(programs, base: dict, var: dict, metric: str, top: int) -> dict:
    ratios, changed_ir, var_fails, base_fails, missing, zero = {}, [], [], [], [], []
    for p in programs:
        case = p["case"]
        b, v = base.get(case), var.get(case)
        if b is None or v is None:
            missing.append(case)
            continue
        if "error" in b:
            base_fails.append(case)
            continue
        if "error" in v:
            var_fails.append((case, v["error"][:120]))
            continue
        bs, vs = b.get(metric), v.get(metric)
        if bs == 0:
            zero.append((case, vs))
            continue
        ratios[case] = vs / bs
        if vs == bs and b.get("ir") != v.get("ir"):
            changed_ir.append(case)
    rq2 = {p["case"] for p in programs if p["rq2"]}
    grown = sorted(((c, r) for c, r in ratios.items() if r > 1), key=lambda x: -x[1])
    shrunk = sorted(((c, r) for c, r in ratios.items() if r < 1), key=lambda x: x[1])
    return {
        "n_all": len(ratios), "gmean_all": gmean(list(ratios.values())),
        "n_rq2": len(ratios.keys() & rq2), "gmean_rq2": gmean([r for c, r in ratios.items() if c in rq2]),
        "changed": len(grown) + len(shrunk) + sum(vs != 0 for _, vs in zero),
        "changed_rq2": sum(c in rq2 for c, _ in grown + shrunk),
        "grown": len(grown), "shrunk": len(shrunk),
        "ir_only": changed_ir, "zero_final": zero,
        "largest": [(c, round(r, 3)) for c, r in grown[:top]],
        "smallest": [(c, round(r, 3)) for c, r in shrunk[:3]],
        "variant_fails": var_fails, "final_fails": base_fails, "missing": missing,
        "ratios": {c: round(r, 6) for c, r in ratios.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--metric", default="size", help="size (default) or poly_cost (halo2 only)")
    ap.add_argument("--json", type=Path, help="write the full summary (with every ratio) here")
    args = ap.parse_args()

    programs, results = load(args.out)
    if "final" not in results:
        sys.exit("no final results: nothing to compare against")
    n_rq2 = sum(p["rq2"] for p in programs)
    print(f"{len(programs)} programs ({n_rq2} compared in RQ2). Ratio = variant / final "
          f"({args.metric}); geomeans over programs where both compile.\n")
    for backend in ("halo2", "noir"):
        base = results["final"].get(backend, {})
        fails = [c for c, r in base.items() if "error" in r]
        print(f"## {backend}: final measured {len(base)}, failed {len(fails)} {fails}")
        print("| variant | gmean 214 (n) | gmean 172 (n) | changed (172) | up / down | IR-only | "
              "variant fails | largest increases |")
        print("|---|---|---|---|---|---|---|---|")
        for variant in results:
            if variant == "final" or backend not in results[variant]:
                continue
            metric = args.metric if backend == "halo2" else "size"
            s = compare(programs, base, results[variant][backend], metric, args.top)
            largest = ", ".join(f"{short(c)} ×{r:.2f}" for c, r in s["largest"]) or "none"
            fails = ", ".join(short(c) for c, _ in s["variant_fails"]) or "none"
            print(f"| {variant} | {s['gmean_all']:.4f} ({s['n_all']}) | {s['gmean_rq2']:.4f} "
                  f"({s['n_rq2']}) | {s['changed']} ({s['changed_rq2']}) | {s['grown']} / "
                  f"{s['shrunk']} | {len(s['ir_only'])} | {fails} | {largest} |")
            if s["missing"]:
                print(f"|   (not measured yet: {len(s['missing'])}) |||||||")
            results[variant][f"_summary_{backend}"] = s
        print()
    if args.json:
        summary = {v: {k[len("_summary_"):]: s for k, s in by.items() if k.startswith("_summary_")}
                   for v, by in results.items() if v != "final"}
        args.json.write_text(json.dumps(summary, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
