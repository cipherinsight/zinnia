"""Analysis of the remote runs: Zinnia against the zkVMs per cap, and Zinnia against itself
with the backend's own array lowering (the delegate systems, `@delegate`).

Inputs: the Zinnia epochs (stage B of `remote/run.sh`: zinnia_halo2,
zinnia_halo2_delegate, zinnia_noir and zinnia_noir_delegate in one run, every
cap and shard) and the zkVM epochs (stage C: sp1 and risc0),
optionally the CAP-16 epochs of the performance set (`--reference-zinnia`,
`--reference-zkvm`, the same measurement at the acceptance study's sizes), and
the count files of stage A (`ablations/measure.py` with configurations `final`
and `fix_delegate`).

Per cap, over that cap's scaled points (an unchanged point is the program of
an earlier cap and is not measured again):

* completion per system: how many points finished (`ok`) and how the others
  ended (failure class: timeout, out_of_memory, ...); nothing is dropped;
* for every pair (Zinnia backend, zkVM) and both clocks (prove_core_s,
  end_to_end_s): per point the ratio median(Zinnia samples) / zkVM sample (the
  zkVMs ran once), and over the points where both completed the geometric mean
  with the harness's bootstrap 95 % interval (`harness.analysis.stats`) and
  the number of points, the copy-only points (`data_movement_only`) apart, as
  `eval-redo/tools/pooled.py` and the harness do;
* each Zinnia backend against its delegate system. The delegate system is timed
  only at (point, target) pairs whose circuit differs from Zinnia's (stage-A
  digests: halo2 IR, Noir main.nr; rule fixed before any timing); at the other
  pairs it compiles the identical circuit and the ratio is 1 by construction.
  Reported: the number of identical and differing pairs, the geometric mean over
  the differing (measured) pairs, and over all pairs;
* sizes: halo2-base cells and Noir gates of Zinnia (`final`) over those of
  `@delegate` (`fix_delegate`) per point, their geometric mean, and how many
  points are cheaper, equal, costlier.

Writes `OUT/scaling_npb.json` and `OUT/scaling_npb.md` (default OUT: DATA/analysis).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from benchmarking.harness.analysis import aggregate, stats
from benchmarking.harness.protocol import OK

ZINNIA = ("zinnia_halo2", "zinnia_noir")
#: Each Zinnia backend's delegate system (the backend's own array lowering).
DELEGATE = {"zinnia_halo2": "zinnia_halo2_delegate", "zinnia_noir": "zinnia_noir_delegate"}
ZKVMS = ("sp1", "risc0")
#: The size each target is counted in (`ablations/measure.py` record fields).
SIZE_FIELD = {"halo2": "cells", "noir": "gates"}
CLOCKS = ("prove_core_s", "end_to_end_s")
REFERENCE_CAP = 16


def case_sets(data: Path) -> dict[int, list[dict]]:
    out = {}
    for path in sorted((data / "case_sets").glob("npbench_pythran_cap*.json")):
        cap = int(path.stem.rsplit("cap", 1)[1])
        out[cap] = json.loads(path.read_text())["cases"]
    return out


def medians(data: aggregate.EpochData, system: str, clock: str) -> dict[str, float]:
    out = {}
    for case, record in data.records.get(system, {}).items():
        values = aggregate.clock_values(record, clock)
        if record["status"] == OK and values:
            out[case] = stats.median(values)
    return out


def summarize(ratios: dict[str, float]) -> dict:
    values = list(ratios.values())
    if not values:
        return {"n": 0}
    low, high = stats.bootstrap_geomean_interval(values)
    return {"n": len(values), "geomean": stats.geometric_mean(values), "ci95": [low, high],
            "zinnia_faster": sum(v < 1 for v in values), "zinnia_slower": sum(v > 1 for v in values)}


def completion(epochs: aggregate.EpochData | None, systems: tuple[str, ...], cases: list[str]) -> dict:
    out = {}
    for system in systems:
        records = epochs.records.get(system, {}) if epochs else {}
        status = {c: records[c]["status"] if c in records else "not_run" for c in cases}
        statuses = Counter(status.values())
        out[system] = {"points": len(cases), "ok": statuses.get(OK, 0),
                       "not_ok": {k: v for k, v in sorted(statuses.items()) if k != OK},
                       "failures": {c: s for c, s in sorted(status.items()) if s != OK}}
    return out


def time_comparison(numerator: aggregate.EpochData, denominator: aggregate.EpochData,
                    pairs: list[tuple[str, str]], cases: list[str], copy_only: set[str]) -> dict:
    """Per pair (system of `numerator`, system of `denominator`) and clock: the ratios of
    median times on the points both completed, summarized."""
    result = {}
    for clock in CLOCKS:
        for top, bottom in pairs:
            a, b = medians(numerator, top, clock), medians(denominator, bottom, clock)
            ratios = {c: a[c] / b[c] for c in cases if c in a and c in b}
            result[f"{top}/{bottom}/{clock}"] = {
                "headline": summarize({c: r for c, r in ratios.items() if c not in copy_only}),
                "copy_only": summarize({c: r for c, r in ratios.items() if c in copy_only}),
                "per_point": ratios}
    return result


def zinnia_vs_zkvm(zinnia, zkvm, cases, copy_only) -> dict:
    return time_comparison(zinnia, zkvm, [(z, k) for z in ZINNIA for k in ZKVMS], cases, copy_only)


def zinnia_vs_delegate(zinnia, cases: list[str], copy_only: set[str], identical: dict[str, set[str]]) -> dict:
    """Zinnia / its delegate system, per target. A (point, target) pair whose delegate circuit is
    identical to Zinnia's (`identical`, from the stage-A digests) is not timed for the delegate:
    its ratio is 1 by construction where Zinnia completed. Reported: the measured (differing)
    pairs alone, and all pairs with the identical ones at 1."""
    measured = time_comparison(zinnia, zinnia, list(DELEGATE.items()), cases, copy_only)
    result = {}
    for key, comparison in measured.items():
        system = key.split("/")[0]
        target = "halo2" if system == "zinnia_halo2" else "noir"
        clock = key.split("/")[2]
        same = identical.get(target, set())
        done = medians(zinnia, system, clock)
        measured_ratios = {c: r for c, r in comparison["per_point"].items() if c not in same}
        by_construction = {c: 1.0 for c in cases if c in same and c in done}
        everything = {**measured_ratios, **by_construction}
        result[key] = {
            "identical_pairs": len([c for c in cases if c in same]),
            "differing_pairs": len([c for c in cases if c not in same]),
            "headline": summarize({c: r for c, r in measured_ratios.items() if c not in copy_only}),
            "copy_only": summarize({c: r for c, r in measured_ratios.items() if c in copy_only}),
            "all_pairs": summarize({c: r for c, r in everything.items() if c not in copy_only}),
            "per_point": everything}
    return result


def identical_pairs(counts: dict[str, dict[str, dict]], cases: list[str]) -> dict[str, set[str]]:
    final, delegate = counts.get("final", {}), counts.get("fix_delegate", {})
    out: dict[str, set[str]] = {}
    for target, key in CIRCUIT_DIGEST.items():
        out[target] = {c for c in cases
                       if (final.get(c, {}).get(target) or {}).get(key)
                       and final[c][target][key] == (delegate.get(c, {}).get(target) or {}).get(key)}
    return out


def load_counts(paths: list[Path]) -> dict[str, dict[str, dict]]:
    """config name -> case -> target -> record, over every case set in the files."""
    out: dict[str, dict[str, dict]] = {}
    for path in paths:
        document = json.loads(path.read_text())
        config = document["config"]["name"]
        for set_name, by_case in document.items():
            if set_name != "config":
                out.setdefault(config, {}).update(by_case)
    return out


def size_comparison(counts: dict[str, dict[str, dict]], cases: list[str]) -> dict:
    """Per target: Zinnia's size (`final`) over the delegate size (`fix_delegate`)."""
    final, delegate = counts.get("final", {}), counts.get("fix_delegate", {})
    out = {}
    for target, field in SIZE_FIELD.items():
        rows, ratios = {}, {}
        for case in cases:
            mine = (final.get(case, {}).get(target) or {})
            theirs = (delegate.get(case, {}).get(target) or {})
            rows[case] = {"zinnia": mine.get(field), "delegate": theirs.get(field),
                          "errors": {name: r["error"][:300] for name, r in (("final", mine), ("fix_delegate", theirs))
                                     if "error" in r}}
            if mine.get(field) and theirs.get(field):
                ratios[case] = mine[field] / theirs[field]
        values = list(ratios.values())
        summary = {"n": len(values), "points": len(cases)}
        if values:
            summary.update(geomean=stats.geometric_mean(values), cheaper=sum(v < 1 for v in values),
                           equal=sum(v == 1 for v in values), costlier=sum(v > 1 for v in values))
        out[target] = {"field": field, "summary": summary, "per_point": rows}
    return out


#: The digest that identifies a target's circuit in a count record (`ablations/measure.py`).
CIRCUIT_DIGEST = {"halo2": "ir", "noir": "noir"}
#: The delegate system of each compile target.
DELEGATE_OF_TARGET = {"halo2": "zinnia_halo2_delegate", "noir": "zinnia_noir_delegate"}


def differing_cases(case_set: Path, count_files: list[Path], target: str) -> list[str]:
    """The case set's scaled cases whose `@delegate` circuit for `target` differs from Zinnia's
    (halo2: IR digest; Noir: main.nr digest), or whose identity is not established (a count
    failed or is missing). Only these are timed for the delegate system; elsewhere it compiles
    the same circuit."""
    counts = load_counts(count_files)
    final, delegate = counts.get("final", {}), counts.get("fix_delegate", {})
    key = CIRCUIT_DIGEST[target]
    scaled = [e["case"] for e in json.loads(case_set.read_text())["cases"] if e["status"] == "perf"]
    out = []
    for case in scaled:
        mine = final.get(case, {}).get(target, {})
        theirs = delegate.get(case, {}).get(target, {})
        if not mine.get(key) or not theirs.get(key) or mine[key] != theirs[key]:
            out.append(case)
    return out


def load(dirs: list[Path] | None) -> aggregate.EpochData | None:
    return aggregate.load_shards(dirs) if dirs else None


def analyze(data: Path, zinnia_dirs: list[Path], zkvm_dirs: list[Path], count_files: list[Path],
            reference_zinnia: list[Path] | None = None, reference_zkvm: list[Path] | None = None) -> dict:
    zinnia, zkvm = load(zinnia_dirs), load(zkvm_dirs)
    counts = load_counts(count_files)
    report: dict = {"caps": {}}
    for cap, entries in sorted(case_sets(data).items()):
        scaled = [e["case"] for e in entries if e["status"] == "perf"]
        copy_only = {e["case"] for e in entries if e.get("data_movement_only")}
        groups = {g: [e["case"] for e in entries if e["status"] == "perf" and e["group"] == g]
                  for g in ("npbench", "pythran")}
        identical = identical_pairs(counts, scaled)
        timed_delegate = {system: [c for c in scaled if c not in identical.get(target, set())]
                          for target, system in DELEGATE_OF_TARGET.items()}
        report["caps"][cap] = {
            "points": {"scaled": len(scaled), **Counter(e["status"] for e in entries if e["status"] != "perf")},
            # a delegate system is timed only where its circuit differs (identical pairs: ratio 1)
            "completion": {**completion(zinnia, ZINNIA, scaled),
                           **{s: completion(zinnia, (s,), cs)[s] for s, cs in timed_delegate.items()},
                           **completion(zkvm, ZKVMS, scaled)},
            "time": zinnia_vs_zkvm(zinnia, zkvm, scaled, copy_only) if zinnia and zkvm else {},
            "time_by_group": {g: zinnia_vs_zkvm(zinnia, zkvm, cs, copy_only) for g, cs in groups.items()}
            if zinnia and zkvm else {},
            "identical_delegate_pairs": {t: len(v) for t, v in identical.items()},
            "time_vs_delegate": zinnia_vs_delegate(zinnia, scaled, copy_only, identical)
            if zinnia else {},
            "size_vs_delegate": size_comparison(counts, scaled) if counts else {},
        }
    if reference_zinnia and reference_zkvm:
        ref_z, ref_k = load(reference_zinnia), load(reference_zkvm)
        cases = sorted(set(ref_z.records.get("zinnia_halo2", {})) | set(ref_z.records.get("zinnia_noir", {})))
        copy_only = aggregate.data_movement_cases(ref_z)
        report["caps"][REFERENCE_CAP] = {"points": {"scaled": len(cases)},
                                         "completion": {**completion(ref_z, ZINNIA, cases),
                                                        **completion(ref_k, ZKVMS, cases)},
                                         "time": zinnia_vs_zkvm(ref_z, ref_k, cases, copy_only),
                                         "size_vs_delegate": size_comparison(counts, cases) if counts else {}}
    return report


def fmt(summary: dict) -> str:
    if not summary.get("n"):
        return "n = 0"
    low, high = summary["ci95"]
    return (f"{summary['geomean']:.3g} [{low:.3g}, {high:.3g}] n = {summary['n']} "
            f"(faster {summary['zinnia_faster']}, slower {summary['zinnia_slower']})")


def time_table(comparisons: dict) -> list[str]:
    lines = ["| ratio | clock | geomean [95 % CI], points (headline) | copy-only |", "|---|---|---|---|"]
    for key, comparison in comparisons.items():
        top, bottom, clock = key.split("/")
        lines.append(f"| {top} / {bottom} | {clock} | {fmt(comparison['headline'])} | "
                     f"{fmt(comparison['copy_only'])} |")
    return lines


def markdown(report: dict) -> str:
    lines = ["# NPBench/Pythran scaling study", ""]
    for cap, part in sorted(report["caps"].items(), key=lambda kv: int(kv[0])):
        lines += [f"## CAP {cap}", "", f"Points: {part['points']}", "", "| system | ok | not ok |", "|---|---|---|"]
        for system, done in part["completion"].items():
            lines.append(f"| {system} | {done['ok']}/{done['points']} | {done['not_ok'] or ''} |")
        if part.get("time"):
            lines += ["", "Zinnia / zkVM:", "", *time_table(part["time"])]
        if part.get("time_vs_delegate"):
            lines += ["", "Zinnia / delegate (identical circuits at 1 by construction):", "",
                      "| ratio | clock | identical / differing pairs | differing pairs (measured) | all pairs |",
                      "|---|---|---|---|---|"]
            for key, c in part["time_vs_delegate"].items():
                top, bottom, clock = key.split("/")
                lines.append(f"| {top} / {bottom} | {clock} | {c['identical_pairs']} / {c['differing_pairs']} | "
                             f"{fmt(c['headline'])} | {fmt(c['all_pairs'])} |")
        for target, sizes in (part.get("size_vs_delegate") or {}).items():
            summary = sizes["summary"]
            if summary.get("n"):
                lines += ["", f"{target} {sizes['field']}, Zinnia / @delegate: geomean {summary['geomean']:.3g} over "
                          f"{summary['n']} of {summary['points']} points (cheaper {summary['cheaper']}, equal "
                          f"{summary['equal']}, costlier {summary['costlier']})."]
        lines.append("")
    return "\n".join(lines)


def main(data: Path, zinnia_dirs: list[Path], zkvm_dirs: list[Path], count_files: list[Path],
         out: Path | None, reference_zinnia: list[Path] | None = None,
         reference_zkvm: list[Path] | None = None) -> int:
    out = out or data / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    report = analyze(data, zinnia_dirs, zkvm_dirs, count_files, reference_zinnia, reference_zkvm)
    (out / "scaling_npb.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    text = markdown(report)
    (out / "scaling_npb.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover - use `python -m benchmarking.scaling_npb analyze`
    raise SystemExit("use: python -m benchmarking.scaling_npb analyze ...")
