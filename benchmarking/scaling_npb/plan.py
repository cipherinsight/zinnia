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


# ---------------------------------------------------------------------------
# The per-step timeout of a cap, and an estimate of stage B's wall time
# ---------------------------------------------------------------------------

#: Seconds of proving per halo2-base cell (prove_core) and per Noir gate (end to end), on 8
#: pinned cores of the remote: the medians over the CAP-16 NPBench/Pythran points above 2e5 cells
#: (1e5 gates) of the fin-zn sessions (2026-10-08, 4115cecb). For planning only.
PROVE_S_PER_UNIT = {"halo2": 9.3e-6, "noir": 1.33e-5}
SIZE_FIELD = {"halo2": "cells", "noir": "gates"}
#: The smallest per-step timeout (the NPBench/Pythran performance sessions used 14,400 s).
TIMEOUT_FLOOR_S = 14400
#: The per-step timeout is at least this many times compile + prove of the slowest point.
TIMEOUT_FACTOR = 3


def step_costs(counts: list[Path]) -> dict[str, dict[str, dict[str, float]]]:
    """case -> configuration -> target -> {"compile_s", "prove_s"} (estimated prove), from the
    stage-A records that completed."""
    from benchmarking.scaling_npb.analyze import load_counts
    out: dict = {}
    for config, by_case in load_counts(counts).items():
        for case, by_target in by_case.items():
            for target, record in by_target.items():
                if "error" in record or record.get(SIZE_FIELD[target]) is None:
                    continue
                out.setdefault(case, {}).setdefault(config, {})[target] = {
                    "compile_s": record["compile_s"],
                    "prove_s": record[SIZE_FIELD[target]] * PROVE_S_PER_UNIT[target]}
    return out


def timeout(counts: list[Path]) -> tuple[int, str]:
    """The cap's per-step timeout: at least TIMEOUT_FACTOR x (compile + estimated prove) of every
    point and configuration whose stage-A count completed, and never below TIMEOUT_FLOOR_S."""
    worst, where = 0.0, "-"
    for case, by_config in step_costs(counts).items():
        for config, by_target in by_config.items():
            for target, cost in by_target.items():
                if cost["compile_s"] + cost["prove_s"] > worst:
                    worst, where = cost["compile_s"] + cost["prove_s"], f"{case} {config} {target}"
    return max(TIMEOUT_FLOOR_S, int(TIMEOUT_FACTOR * worst + 0.5)), f"slowest step {worst:.0f} s ({where})"


def estimate(data: Path, cap: int, shards: int, counts: list[Path], repeats: int, heavy_repeats: int,
             slots: int = 4) -> dict:
    """Stage B's wall time at this cap from the stage-A compile times and estimated prove times:
    a run of R repeats compiles and proves R + 2 times per system (setup, warm-up, R samples);
    verification is not counted. Shards go to `slots` slots in order (the queue)."""
    costs = step_costs(counts)
    unknown: set[str] = set()
    shard_s = []
    for shard in range(shards):
        total = 0.0
        for _, systems, count, cases in runs(data, cap, shard, shards, counts, repeats, heavy_repeats):
            for case in cases:
                for system in systems.split(","):
                    target = "halo2" if "halo2" in system else "noir"
                    config = "fix_delegate" if system.endswith("_delegate") else "final"
                    cost = costs.get(case, {}).get(config, {}).get(target)
                    if cost is None:
                        unknown.add(case)
                        continue
                    total += (count + 2) * (cost["compile_s"] + cost["prove_s"])
        shard_s.append(total)
    finish = [0.0] * slots
    for seconds in shard_s:                    # the queue: a free slot takes the next shard
        finish[finish.index(min(finish))] += seconds
    return {"cap": cap, "shard_hours": [round(s / 3600, 2) for s in shard_s],
            "total_slot_hours": round(sum(shard_s) / 3600, 1), "wall_hours_4_slots": round(max(finish) / 3600, 1),
            "points_without_counts": sorted(unknown)}
