"""The Zinnia timing runs of one shard of one cap (stage B of `remote/run.sh`).

Both rules below were fixed before any timing.

* Systems: zinnia_halo2 and zinnia_noir run on every scaled point of the shard.
  Each delegate system (Zinnia with the backend's own array lowering) runs only
  on the points whose `@delegate` circuit for its target differs from Zinnia's
  in the stage-A counts, or where that could not be established
  (`analyze.differing_cases`); elsewhere it compiles the identical circuit and
  its ratio to Zinnia is 1 by construction.
* Repeats: 5, and 3 (the author's floor) at the points listed in
  `DATA/validate_on_remote.txt`, the points too large or too memory-hungry to
  validate on the 16-core, 46 GB local machine. Every compile step of the
  harness compiles the kernel again, so at these points each repeat costs a
  compile that can take hours. Every system at a point gets the same count.

Shards are the harness's (`case_sets.shard_cases` over the cap's case set), so
the shard of a point does not depend on which runs it is in. Each output line
is one harness run: `<label suffix> <systems> <repeats> <cases>`.
"""

from __future__ import annotations

from pathlib import Path

from benchmarking.harness.case_sets import load_perf_cases, shard_cases
from benchmarking.scaling_npb.analyze import DELEGATE_OF_TARGET, differing_cases
from benchmarking.scaling_npb.generate import case_set_path

ZINNIA = "zinnia_halo2,zinnia_noir"


def heavy_cases(data: Path, cap: int) -> set[str]:
    listing = data / "validate_on_remote.txt"
    if not listing.exists():
        return set()
    rows = (line.split() for line in listing.read_text().splitlines() if line.strip())
    return {case for row_cap, case, _ in rows if int(row_cap) == cap}


def runs(data: Path, cap: int, shard: int, shards: int, counts: list[Path], repeats: int,
         heavy_repeats: int) -> list[tuple[str, str, int, list[str]]]:
    case_set = case_set_path(data, cap)
    mine = [c.case for c in shard_cases(load_perf_cases(case_set, None, None), shard, shards) if c.included]
    heavy = heavy_cases(data, cap)
    plan = []
    for suffix, cases, count in (("", [c for c in mine if c not in heavy], repeats),
                                 ("-heavy", [c for c in mine if c in heavy], heavy_repeats)):
        if cases:
            plan.append((suffix, ZINNIA, count, cases))
        for target, system in DELEGATE_OF_TARGET.items():
            differing = set(differing_cases(case_set, counts, target))
            chosen = [c for c in cases if c in differing]
            if chosen:
                plan.append((f"{suffix}-{system}", system, count, chosen))
    return plan


def main(data: Path, cap: int, shard: int, shards: int, counts: list[Path], repeats: int,
         heavy_repeats: int) -> int:
    for suffix, systems, count, cases in runs(data, cap, shard, shards, counts, repeats, heavy_repeats):
        print(suffix or "-", systems, count, ",".join(cases))
    return 0
