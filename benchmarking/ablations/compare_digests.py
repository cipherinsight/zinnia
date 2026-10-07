#!/usr/bin/env python3
"""Compare two `benchmarking/determinism.py --runs 1` outputs case by case.

    python benchmarking/ablations/compare_digests.py BEFORE.json AFTER.json

For every case and target present in either file: the IR digest (`ir`), the
emitted Noir source digest (`noir`, Noir target), halo2-base cells and `k`
(halo2 target), and any error. Prints the cases that differ and a one-line
summary; exit status 0 iff every case and target is identical.
"""

from __future__ import annotations

import json
import sys

KEYS = ("ir", "noir", "cells", "k", "error", "backend_error")


def record(report: dict, case: str, target: str) -> dict | None:
    runs = report.get(case, {}).get(target, {}).get("runs")
    return {k: runs[0].get(k) for k in KEYS} if runs else None


def main(before_path: str, after_path: str) -> int:
    before, after = (json.load(open(p)) for p in (before_path, after_path))
    pairs = sorted({(c, t) for r in (before, after) for c, per in r.items() for t in per})
    differ = [(c, t) for c, t in pairs if record(before, c, t) != record(after, c, t)]
    for case, target in differ:
        print(f"DIFFERS {case} {target}: {record(before, case, target)} -> {record(after, case, target)}")
    cases = len({c for c, _ in pairs})
    print(f"{len(pairs) - len(differ)}/{len(pairs)} case-targets identical over {cases} cases "
          f"(IR digest, Noir source digest, halo2 cells and k)")
    return 1 if differ else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
