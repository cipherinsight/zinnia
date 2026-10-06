"""Summaries of one epoch, computed only from its directory (or from the
directories of the shards of one run, `run --shard K/N`, read together).

Writes `summary/` inside the epoch (or the directory given for shards):

- `per_case.csv`: one row per (system, case): status, sample count, median and
  IQR of each clock, proof size;
- `summary.json` and `summary.md`:
  - coverage: for every system, how many cases it completed and why the others
    failed (performance is reported separately from expressiveness);
  - the all-supported subset: the cases every system of the epoch completed,
    and on it, for every (Zinnia system, baseline) pair and every clock, the
    geometric mean of per-case median ratios (Zinnia / baseline; below 1 means
    Zinnia is faster) with a bootstrap 95 % interval;
  - the same pairs on each pair's own common subset;
  - when the case set labels cases with a `group` (e.g. the corpus), the
    headline comparison within each group;
  - every excluded case with its reason;
  - the acceptance table, when the epoch has `acceptance/results.json`.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from benchmarking.harness.adapters import BASELINE_SYSTEMS, SAME_PROOF_SYSTEM, ZINNIA_SYSTEMS
from benchmarking.harness.analysis import stats
from benchmarking.harness.case_sets import INCLUDED
from benchmarking.harness.protocol import OK, REFERENCE_SYSTEM

CLOCKS = ("prove_core_s", "end_to_end_s", "verify_s")


@dataclass
class EpochData:
    root: Path
    manifest: dict
    case_set: dict
    records: dict[str, dict[str, dict]]  # system -> case -> raw record

    @property
    def systems(self) -> list[str]:
        order = list(ZINNIA_SYSTEMS) + list(BASELINE_SYSTEMS)
        present = [s for s in order if s in self.records]
        return present + sorted(set(self.records) - set(present))


def load_epoch(root: Path) -> EpochData:
    records: dict[str, dict[str, dict]] = {}
    raw = root / "raw"
    for system_dir in sorted(p for p in raw.iterdir() if p.is_dir()) if raw.exists() else []:
        if system_dir.name == REFERENCE_SYSTEM:
            continue
        records[system_dir.name] = {p.stem: json.loads(p.read_text())
                                    for p in sorted(system_dir.glob("*.json"))}
    manifest = json.loads((root / "manifest.json").read_text())
    case_set_path = root / "case_set.json"
    case_set = json.loads(case_set_path.read_text()) if case_set_path.exists() else {"cases": []}
    return EpochData(root, manifest, case_set, records)


def load_shards(roots: list[Path]) -> EpochData:
    """The epochs of the shards of one run as one epoch.

    Every (system, case) record must come from exactly one shard. The case set
    is the union of the shards' (each lists the excluded cases too); the
    manifest names the shards and keeps each one's manifest.
    """
    shards = [load_epoch(root) for root in roots]
    records: dict[str, dict[str, dict]] = {}
    for shard in shards:
        for system, by_case in shard.records.items():
            merged = records.setdefault(system, {})
            twice = sorted(set(merged) & set(by_case))
            if twice:
                raise ValueError(f"{system}: cases {twice} appear in more than one shard")
            merged.update(by_case)
    cases: dict[str, dict] = {}
    for shard in shards:
        for entry in _case_entries(shard):
            cases.setdefault(entry["case"], entry)
    manifest = {"epoch": " + ".join(s.manifest.get("epoch", s.root.name) for s in shards),
                "shards": [s.manifest for s in shards]}
    return EpochData(roots[0], manifest, {"cases": list(cases.values())}, records)


def clock_values(record: dict, clock: str) -> list[float]:
    """The timed samples of one clock (warm-up excluded)."""
    values = []
    for sample in record["samples"]:
        if clock == "verify_s":
            result = (sample.get("verify") or {}).get("result") or {}
            value = result.get("verify_s")
        else:
            value = (sample.get("prove") or {}).get(clock)
        if value is not None:
            values.append(float(value))
    return values


def setup_facts(record: dict) -> dict:
    """Circuit size facts from setup: halo2 `k`/`lookup_bits`, Noir circuit size."""
    details = ((record.get("setup") or {}).get("result") or {}).get("details") or {}
    params = details.get("params") or details
    gates = details.get("gates")
    facts = {"k": params.get("k"), "lookup_bits": params.get("lookup_bits"),
             "circuit_size": gates.get("circuit_size") if isinstance(gates, dict) else gates}
    return {key: value for key, value in facts.items() if value is not None}


def per_case_rows(data: EpochData) -> list[dict]:
    rows = []
    for system in data.systems:
        for case, record in sorted(data.records[system].items()):
            row = {"system": system, "case": case, "status": record["status"],
                   "samples": len(record["samples"])}
            for clock in CLOCKS:
                values = clock_values(record, clock)
                if values:
                    q1, q3 = stats.quartiles(values)
                    row.update({f"{clock}_median": stats.median(values),
                                f"{clock}_q1": q1, f"{clock}_q3": q3})
            sizes = [s["prove"]["proof_bytes"] for s in record["samples"] if s.get("prove")]
            row["proof_bytes"] = sizes[0] if sizes else None
            row.update(setup_facts(record))
            rows.append(row)
    return rows


def coverage(data: EpochData) -> dict:
    out = {}
    for system in data.systems:
        statuses = Counter(r["status"] for r in data.records[system].values())
        failures = [{"case": case, "class": r["failure"]["class"], "step": r["failure"]["step"],
                     "detail": r["failure"]["detail"][:300]}
                    for case, r in sorted(data.records[system].items()) if r["failure"]]
        out[system] = {"cases": len(data.records[system]), "ok": statuses.get(OK, 0),
                       "by_status": dict(statuses), "failures": failures}
    return out


def supported(data: EpochData, system: str) -> set[str]:
    return {case for case, r in data.records[system].items() if r["status"] == OK}


def common_subset(data: EpochData, systems: list[str]) -> list[str]:
    sets = [supported(data, s) for s in systems]
    return sorted(set.intersection(*sets)) if sets else []


def compare(data: EpochData, zinnia: str, baseline: str, cases: list[str],
            clock: str) -> dict:
    """Geomean of per-case median ratios zinnia/baseline on `cases`."""
    ratios = []
    per_case = {}
    for case in cases:
        z = clock_values(data.records[zinnia][case], clock)
        b = clock_values(data.records[baseline][case], clock)
        if not z or not b:
            continue
        ratio = stats.median(z) / stats.median(b)
        ratios.append(ratio)
        per_case[case] = ratio
    result = {"zinnia": zinnia, "baseline": baseline, "clock": clock, "n": len(ratios),
              "per_case_ratio": per_case}
    if ratios:
        low, high = stats.bootstrap_geomean_interval(ratios)
        result.update(geomean_ratio=stats.geometric_mean(ratios), ci95=[low, high],
                      zinnia_faster=sum(r < 1 for r in ratios),
                      zinnia_slower=sum(r > 1 for r in ratios))
    return result


def comparison_pairs(data: EpochData) -> list[tuple[str, str]]:
    """Every (Zinnia system, baseline) pair; same-proof-system pairs first."""
    zinnia = [s for s in data.systems if s in ZINNIA_SYSTEMS]
    baselines = [s for s in data.systems if s not in ZINNIA_SYSTEMS]
    pairs = [(z, b) for z in zinnia for b in baselines]
    return sorted(pairs, key=lambda pair: SAME_PROOF_SYSTEM.get(pair[0]) != pair[1])


def _case_entries(data: EpochData) -> list[dict]:
    return [c if isinstance(c, dict) else {"case": c} for c in data.case_set.get("cases", [])]


def _excluded(entry: dict) -> bool:
    """Whether the case set keeps a case out of the run (any status but `perf`)."""
    if "status" in entry:
        return entry["status"] != INCLUDED
    return not entry.get("included", True)


def exclusions(data: EpochData) -> list[dict]:
    return [{"case": c["case"], "reason": c.get("reason", c.get("status"))}
            for c in _case_entries(data) if _excluded(c)]


def data_movement_cases(data: EpochData) -> set[str]:
    """Cases that only copy inputs; reported apart from the headline comparison."""
    return {c["case"] for c in _case_entries(data) if c.get("data_movement_only")}


def case_groups(data: EpochData) -> dict[str, set[str]]:
    """The cases of each `group` label of the case set (none: no groups)."""
    groups: dict[str, set[str]] = {}
    for entry in _case_entries(data):
        if entry.get("group"):
            groups.setdefault(entry["group"], set()).add(entry["case"])
    return groups


def acceptance_summary(root: Path) -> dict | None:
    path = root / "acceptance" / "results.json"
    if not path.exists():
        return None
    results = json.loads(path.read_text())["results"]
    table: dict[str, Counter] = {}
    for row in results:
        table.setdefault(row["dataset"], Counter())[row["outcome"]] += 1
    return {dataset: dict(counts) for dataset, counts in sorted(table.items())}


def summarize(epochs: Path | list[Path], out: Path | None = None) -> dict:
    """Write the summary of one epoch (or of the shards of one run) to `out`
    (default: `<epoch>/summary`) and return it."""
    roots = [epochs] if isinstance(epochs, Path) else list(epochs)
    data = load_epoch(roots[0]) if len(roots) == 1 else load_shards(roots)
    all_supported = common_subset(data, data.systems)
    movement = data_movement_cases(data)
    headline = [c for c in all_supported if c not in movement]
    movement_only = [c for c in all_supported if c in movement]
    pairs = comparison_pairs(data)
    summary = {
        "epoch": data.manifest.get("epoch"),
        "systems": data.systems,
        "coverage": coverage(data),
        "all_supported_subset": all_supported,
        "headline_subset": headline,
        "data_movement_subset": movement_only,
        "comparisons_headline": [compare(data, z, b, headline, clock)
                                 for z, b in pairs for clock in CLOCKS],
        "comparisons_data_movement": [compare(data, z, b, movement_only, clock)
                                      for z, b in pairs for clock in CLOCKS],
        "comparisons_all_supported": [compare(data, z, b, all_supported, clock)
                                      for z, b in pairs for clock in CLOCKS],
        "comparisons_pairwise": [compare(data, z, b, common_subset(data, [z, b]), clock)
                                 for z, b in pairs for clock in CLOCKS],
        "comparisons_by_group": {
            group: {"headline_subset": [c for c in headline if c in members],
                    "comparisons": [compare(data, z, b, [c for c in headline if c in members],
                                            clock) for z, b in pairs for clock in CLOCKS]}
            for group, members in sorted(case_groups(data).items())},
        "excluded_cases": exclusions(data),
        "acceptance": acceptance_summary(roots[0]),
    }
    out = out or roots[0] / "summary"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    _write_csv(out / "per_case.csv", per_case_rows(data))
    (out / "summary.md").write_text(render_markdown(summary, data))
    return summary


def _write_csv(path: Path, rows: list[dict]) -> None:
    columns = sorted({key for row in rows for key in row},
                     key=lambda k: (k not in ("system", "case", "status", "samples"), k))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _comparison_lines(comparisons: list[dict]) -> list[str]:
    lines = ["| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |",
             "|---|---|---|---|---|---|---|"]
    for c in comparisons:
        if c["n"] == 0:
            lines.append(f"| {c['zinnia']} | {c['baseline']} | {c['clock']} | 0 | - | - | - |")
            continue
        lines.append(f"| {c['zinnia']} | {c['baseline']} | {c['clock']} | {c['n']} | "
                     f"{c['geomean_ratio']:.3f} | [{c['ci95'][0]:.3f}, {c['ci95'][1]:.3f}] | "
                     f"{c['zinnia_faster']} / {c['zinnia_slower']} |")
    return lines


def render_markdown(summary: dict, data: EpochData) -> str:
    lines = [f"# Epoch {summary['epoch']}", "",
             "Ratios are Zinnia / baseline of per-case medians; below 1 means Zinnia is faster.",
             "Only cases with status `ok` (proved, verified, outputs equal to NumPy, tampered "
             "proof rejected) enter a comparison.", "", "## Coverage", "",
             "| system | cases | ok | other statuses |", "|---|---|---|---|"]
    for system, cov in summary["coverage"].items():
        others = ", ".join(f"{k}: {v}" for k, v in sorted(cov["by_status"].items()) if k != OK)
        lines.append(f"| {system} | {cov['cases']} | {cov['ok']} | {others or '-'} |")
    headline = summary["headline_subset"]
    lines += ["", "## Headline: all-supported subset without copy-only cases", "",
              f"{len(headline)} cases completed by every system that do more than copy "
              f"inputs: {', '.join(headline) or '(none)'}", ""]
    lines += _comparison_lines(summary["comparisons_headline"])
    movement = summary["data_movement_subset"]
    lines += ["", "## Copy-only cases (reported apart)", "",
              f"{len(movement)} all-supported cases that only copy inputs: "
              f"{', '.join(movement) or '(none)'}", ""]
    lines += _comparison_lines(summary["comparisons_data_movement"])
    for group, by_group in summary["comparisons_by_group"].items():
        members = by_group["headline_subset"]
        lines += ["", f"## Group {group}: headline cases", "",
                  f"{len(members)} headline cases of group {group}: "
                  f"{', '.join(members) or '(none)'}", ""]
        lines += _comparison_lines(by_group["comparisons"])
    subset = summary["all_supported_subset"]
    lines += ["", "## All-supported subset", "",
              f"{len(subset)} cases completed by every system of the epoch "
              f"({', '.join(summary['systems'])}): {', '.join(subset) or '(none)'}", ""]
    lines += _comparison_lines(summary["comparisons_all_supported"])
    lines += ["", "## Pairwise subsets", "",
              "Each pair on the cases both of its systems completed.", ""]
    lines += _comparison_lines(summary["comparisons_pairwise"])
    lines += ["", "## Excluded cases", ""]
    excluded = summary["excluded_cases"]
    lines += [f"- {e['case']}: {e['reason']}" for e in excluded] or ["(none)"]
    lines += ["", "## Failures by system", ""]
    for system, cov in summary["coverage"].items():
        for f in cov["failures"]:
            lines.append(f"- {system} {f['case']}: {f['class']} at {f['step']}: "
                         f"{f['detail'][:160]}")
    if summary["acceptance"]:
        lines += ["", "## Acceptance", "", "| dataset | outcome | programs |", "|---|---|---|"]
        for dataset, counts in summary["acceptance"].items():
            for outcome, count in sorted(counts.items()):
                lines.append(f"| {dataset} | {outcome} | {count} |")
    return "\n".join(lines) + "\n"
