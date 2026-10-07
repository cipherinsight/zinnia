#!/usr/bin/env python3
"""Soundness check of the circuits one compiler configuration builds.

Runs the harness's measurement protocol (`benchmarking/harness/protocol.py`) with one
prove per case, on the Zinnia systems, with the configuration's environment variables
set (`common.CONFIGS`; every step's child process inherits them):

1. setup on input A (compile, halo2-base keys / Noir package and verification key);
2. a proof on input B (halo2-base with the keys of A; Noir with `bb prove`), whose
   public outputs must equal the NumPy reference position by position;
3. verification of that proof;
4. a copy of the proof with its first public value changed, which must be rejected.

A case passes on a system when every step succeeds (raw status `ok`).

    python benchmarking/ablations/soundness.py --config abl_joint --out DIR \\
        --cases ds1000:0345,npbench_pythran:pythran_lu__factorMatrix0 \\
        [--systems zinnia_halo2,zinnia_noir] [--memory-max 16G] [--timeout 1800] [--threads 8]

Writes `DIR/<config>/raw/<system>/<case>.json` (the harness's raw records) and
`DIR/<config>/summary.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

sys.path.insert(0, str(common.ROOT))
from benchmarking.harness.adapters import get_adapter  # noqa: E402
from benchmarking.harness.case_sets import load_perf_cases  # noqa: E402
from benchmarking.harness.protocol import CaseProtocol, Epoch, RunConfig, StepRunner  # noqa: E402


def case_ids(spec: str) -> dict[str, list[str]]:
    """`set:case,...` (a DS-1000 case may be given by its number) by case set."""
    by_set: dict[str, list[str]] = {}
    for item in spec.split(","):
        set_name, _, case = item.partition(":")
        if set_name == "ds1000" and case.isdigit():
            case = f"ds1000_case{int(case):04d}"
        by_set.setdefault(set_name, []).append(case)
    return by_set


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, choices=sorted(common.CONFIGS))
    ap.add_argument("--cases", required=True, help="set:case,... (sets: ds1000, npbench_pythran)")
    ap.add_argument("--systems", default="zinnia_halo2,zinnia_noir")
    ap.add_argument("--memory-max", default="16G")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--threads", type=int, default=8, help="prover threads per step")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    os.environ.update(common.CONFIGS[args.config])
    os.environ.setdefault("ZINNIA_SKIP_RELEASE_BUILD", "1")
    root = args.out / args.config
    config = RunConfig(systems=args.systems.split(","), repeats=1, step_timeout_s=args.timeout,
                       memory_max=None if args.memory_max == "none" else args.memory_max,
                       threads=args.threads)
    protocol = CaseProtocol(StepRunner(Epoch(root), config), config, random.Random(20261008))
    adapters = [get_adapter(name) for name in config.systems]
    summary: dict[str, dict[str, str]] = {}
    for set_name, ids in case_ids(args.cases).items():
        for case in load_perf_cases(common.CASE_SETS[set_name], ids, None):
            for run in protocol.run(case, adapters):
                status = run.record["status"]
                summary.setdefault(case.case, {})[run.adapter.name] = status
                failure = run.record["failure"] or {}
                tamper = (run.record.get("tamper") or {}).get("status")
                print(f"{case.case} {run.adapter.name} {status} tamper={tamper} "
                      f"{failure.get('detail', '')[:300]}", flush=True)
    (root / "summary.json").write_text(json.dumps({"config": args.config,
                                                   "env": common.CONFIGS[args.config],
                                                   "status": summary}, indent=1) + "\n")
    failed = [(c, s, v) for c, per in summary.items() for s, v in per.items() if v != "ok"]
    print(f"{sum(len(p) for p in summary.values()) - len(failed)} of "
          f"{sum(len(p) for p in summary.values())} case-systems ok; not ok: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
