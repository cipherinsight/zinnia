#!/usr/bin/env python3
"""Costs, IR digests and layer decisions of one compiler configuration, per perf case.

For every `perf` case of the case sets and both targets, each compile in its own child
process with the configuration's environment (`common.CONFIGS`, or `--env NAME=VALUE`):
halo2-base cells (and `k`), the Noir circuit's `bb gates` (with `--gates`), the IR and
`main.nr` digests, and what the array-lowering layer chose (`common.summarize_report`):
per family the constructions chosen, the groups lowered through the backend's memory,
and for a named fixed configuration the groups that kept the layer's choice
(fallbacks).

    python benchmarking/ablations/measure.py --config abl_joint --out abl_joint.json \\
        [--case-sets ds1000,npbench_pythran] [--cases 0345,pythran_lu__factorMatrix0] \\
        [--targets halo2,noir] [--gates] [--jobs 8] [--timeout 1200] \\
        [--wrap "python benchmarking/ablations/capped.py 8G none"]

The output maps case set -> case -> target -> record (`common.measure`), with the
configuration and its environment under `"config"`.
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


def parse_env(pairs: list[str]) -> dict[str, str]:
    env = {}
    for pair in pairs:
        name, _, value = pair.partition("=")
        env[name] = value
    return env


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help=f"one of {', '.join(common.CONFIGS)}, or a "
                    "label for --env")
    ap.add_argument("--env", action="append", default=[], help="NAME=VALUE (with a custom label)")
    ap.add_argument("--case-sets", default=",".join(common.CASE_SETS))
    ap.add_argument("--cases", help="comma-separated case names (default: every perf case)")
    ap.add_argument("--targets", default=",".join(common.TARGETS))
    ap.add_argument("--gates", action="store_true", help="also measure Noir bb gates")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=1200)
    ap.add_argument("--wrap", default="", help="command prefix of each child (a memory cap)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.config in common.CONFIGS and args.env:
        ap.error("--env takes a label of its own, not a predefined configuration")
    env = common.CONFIGS.get(args.config) if not args.env else parse_env(args.env)
    if env is None:
        ap.error(f"unknown configuration {args.config}")
    wanted = set(args.cases.split(",")) if args.cases else None
    jobs = []
    for set_name in args.case_sets.split(","):
        entries = common.perf_entries(common.CASE_SETS[set_name])
        for case in sorted(entries):
            short = case.removeprefix("ds1000_case")
            if wanted is None or case in wanted or short in wanted:
                for target in args.targets.split(","):
                    jobs.append((set_name, case, entries[case], target))
    wrap = shlex.split(args.wrap)

    def run(job):
        set_name, case, entry, target = job
        record = common.run_child(entry, target, env, args.gates, args.timeout, wrap)
        cost = common.cost_of(record, target)
        layer = record.get("layer", {})
        print(f"{case} {target} cost={cost} fallback={layer.get('fallback')} "
              f"native={layer.get('native_memory')} {record.get('error', '')[:200]}", flush=True)
        return job, record

    results: dict = {"config": {"name": args.config, "env": env}}
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for (set_name, case, _, target), record in pool.map(run, jobs):
            results.setdefault(set_name, {}).setdefault(case, {})[target] = record
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1, sort_keys=True) + "\n")
    errors = [(s, c, t) for s, per in results.items() if s != "config" for c, by in per.items()
              for t, r in by.items() if "error" in r]
    print(f"{len(jobs) - len(errors)}/{len(jobs)} measured; errors: {errors}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
