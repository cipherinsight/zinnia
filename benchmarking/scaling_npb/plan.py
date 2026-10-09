"""Job lines of the scaling study for the remote job queue (`eval-redo/post-freeze/scripts/queue.sh`),
the per-step timeout of each cap, and an estimate of the timed stage's wall time.

The queue runs one job per line, `timed <command>` or `untimed <command>`, with the slot's CPU
list appended to the command; `remote/job.sh` is the command. Lines are ordered longest first
(by input size for the untimed jobs, by estimated cost for the timed ones), so long jobs do not
start last.

Untimed jobs (stages V and A), one per (point, configuration) or (point, system):

* `count <cap> <case> <config>`: halo2-base cells and Noir gates of every scaled point, as Zinnia
  compiles it (`final`) and with the backend's own array lowering (`fix_delegate`);
* `validate <cap> <case> <system>`: the output check of the four Zinnia systems at the points
  the local machine could not validate (`DATA/validate_on_remote.txt`).

Timed jobs (stage B), one per scaled point, written after stage A because they depend on it:
`time <cap> <case> <repeats> <delegate systems or ->`. Each times zinnia_halo2 and zinnia_noir
and then, in the same slot, the delegate system of each target whose `@delegate` circuit differs
from Zinnia's at that point (`analyze.differing_cases`: halo2 IR digest, Noir main.nr digest, or
a count that failed or is missing). Where the circuit is identical the delegate's ratio is 1 by
construction (rule fixed before any timing, 2026-10-09). Repeats: 5, and 3 (the author's floor)
at the points of `validate_on_remote.txt`, whose compiles are long and are repeated by every
harness step; the same count for every system at a point.
"""

from __future__ import annotations

import json
from pathlib import Path

from benchmarking.scaling_npb.analyze import DELEGATE_OF_TARGET, differing_cases, load_counts
from benchmarking.scaling_npb.generate import case_set_path

ZINNIA_SYSTEMS = ("zinnia_halo2", "zinnia_halo2_delegate", "zinnia_noir", "zinnia_noir_delegate")
CONFIGS = ("final", "fix_delegate")

#: Seconds of proving per halo2-base cell (prove_core) and per Noir gate (end to end), on 8
#: pinned cores of the remote: the medians over the CAP-16 NPBench/Pythran points above 2e5 cells
#: (1e5 gates) of the fin-zn sessions (2026-10-08, 4115cecb). For planning only.
PROVE_S_PER_UNIT = {"halo2": 9.3e-6, "noir": 1.33e-5}
SIZE_FIELD = {"halo2": "cells", "noir": "gates"}
#: The smallest per-step timeout (the NPBench/Pythran performance sessions used 14,400 s).
TIMEOUT_FLOOR_S = 14400
#: The per-step timeout is at least this many times compile + prove of the slowest step.
TIMEOUT_FACTOR = 3


def scaled_points(data: Path, cap: int) -> list[dict]:
    return [e for e in json.loads(case_set_path(data, cap).read_text())["cases"] if e["status"] == "perf"]


def heavy_cases(data: Path, cap: int) -> set[str]:
    """The cap's points the local machine left to the remote (too large or too memory-hungry)."""
    listing = data / "validate_on_remote.txt"
    if not listing.exists():
        return set()
    rows = (line.split() for line in listing.read_text().splitlines() if line.strip())
    return {case for row_cap, case, _ in rows if int(row_cap) == cap}


def untimed_lines(data: Path, caps: list[int], script: str) -> list[str]:
    jobs = []
    for cap in caps:
        heavy = heavy_cases(data, cap)
        for point in scaled_points(data, cap):
            size = point["scaling"]["input_elements"]
            for config in CONFIGS:
                jobs.append((size, f"untimed bash {script} count {cap} {point['case']} {config}"))
            if point["case"] in heavy:
                for system in ZINNIA_SYSTEMS:
                    jobs.append((size, f"untimed bash {script} validate {cap} {point['case']} {system}"))
    return [line for _, line in sorted(jobs, key=lambda job: -job[0])]


def step_costs(counts: list[Path]) -> dict[str, dict[str, dict[str, float]]]:
    """case -> configuration -> target -> {"compile_s", "prove_s"} (estimated prove), from the
    stage-A records that completed."""
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
    point and configuration whose stage-A count completed, and never below TIMEOUT_FLOOR_S. It is
    the same for every system at the cap's points (Zinnia, delegates, zkVMs)."""
    worst, where = 0.0, "-"
    for case, by_config in step_costs(counts).items():
        for config, by_target in by_config.items():
            for target, cost in by_target.items():
                if cost["compile_s"] + cost["prove_s"] > worst:
                    worst, where = cost["compile_s"] + cost["prove_s"], f"{case} {config} {target}"
    return max(TIMEOUT_FLOOR_S, int(TIMEOUT_FACTOR * worst + 0.5)), f"slowest step {worst:.0f} s ({where})"


def timed_jobs(data: Path, cap: int, counts: list[Path], repeats: int,
               heavy_repeats: int) -> list[tuple[float | None, int, str, str]]:
    """(estimated seconds or None, repeats, case, delegate systems) per scaled point. A run of R
    repeats compiles and proves R + 2 times per system (setup, warm-up, R samples)."""
    case_set = case_set_path(data, cap)
    heavy = heavy_cases(data, cap)
    differing = {target: set(differing_cases(case_set, counts, target)) for target in DELEGATE_OF_TARGET}
    costs = step_costs(counts)
    jobs = []
    for point in scaled_points(data, cap):
        case = point["case"]
        count = heavy_repeats if case in heavy else repeats
        delegates = [system for target, system in DELEGATE_OF_TARGET.items() if case in differing[target]]
        steps = [("final", "halo2"), ("final", "noir")]
        steps += [("fix_delegate", "halo2" if "halo2" in system else "noir") for system in delegates]
        known = [costs.get(case, {}).get(config, {}).get(target) for config, target in steps]
        seconds = None if None in known else sum((count + 2) * (c["compile_s"] + c["prove_s"]) for c in known)
        jobs.append((seconds, count, case, ",".join(delegates) or "-"))
    return jobs


def timed_lines(data: Path, caps: list[int], counts_of, repeats: int, heavy_repeats: int,
                script: str) -> list[str]:
    """Timed job lines of the caps, in cap order; within a cap the points without an estimate
    (their counts failed: the largest) first, then by estimated cost, longest first."""
    lines = []
    for cap in caps:
        jobs = timed_jobs(data, cap, counts_of(cap), repeats, heavy_repeats)
        jobs.sort(key=lambda job: (job[0] is not None, -(job[0] or 0)))
        lines += [f"timed bash {script} time {cap} {case} {count} {delegates}"
                  for _, count, case, delegates in jobs]
    return lines


def estimate(data: Path, cap: int, counts: list[Path], repeats: int, heavy_repeats: int, slots: int = 4) -> dict:
    """The timed stage's wall time at this cap on `slots` slots (longest job first to the
    first free slot), over the points with an estimate; the others are listed."""
    jobs = timed_jobs(data, cap, counts, repeats, heavy_repeats)
    known = sorted((seconds for seconds, *_ in jobs if seconds is not None), reverse=True)
    finish = [0.0] * slots
    for seconds in known:
        finish[finish.index(min(finish))] += seconds
    return {"cap": cap, "points": len(jobs), "slot_hours": round(sum(known) / 3600, 1),
            "wall_hours": round(max(finish) / 3600, 1), "slots": slots,
            "points_without_estimate": [case for seconds, _, case, _ in jobs if seconds is None]}
