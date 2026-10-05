"""`revision/phase1/results/REPORT.md`, the report of `acceptance.md` §3.

One pass over the per-circuit cache (`cache.py`), every entry handed to
the seven measurements of `acceptance.md` §2, then one markdown file with
every table and a one-paragraph reading of each, in the shape of
`revision/phase0/results/REPORT.md`:

| Section | Module |
|---|---|
| 1 acceptance and invariants | `invariants.py` |
| 2 graph size against program size | `sizes.py` |
| 3 capacities, three columns | `capacities.py` |
| 4 questions, memo, solver time, unknowns | `questions.py` |
| 5 tightness verification | `tightness.py` |
| 6 baseline path and greedy cost | `selection.py` |
| 7 evidence replay | `replay.py` |
| 8 the acceptance criteria | `acceptance.md` §4 |

The tables the criteria are asserted on are also written to
`<cache>/tables.json`, which `test_acceptance.py` reads, so that the test
and the report see one set of numbers. Two runs over the same cache give the
same file: nothing here reads a clock or draws without a seed.

```sh
.venv/bin/python -m testing.phase1.report [--per-circuit N] [--draws K]
```
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from testing.phase0 import REPO_ROOT
from testing.phase0.report import capped, histogram, table
from testing.phase1 import capacities, invariants, questions, replay, selection, sizes, tightness
from testing.phase1.cache import CACHE_DIR, Entry, entries, missing, size_bytes

RESULTS = REPO_ROOT.parent / "revision" / "phase1" / "results"
REPORT_MD = RESULTS / "REPORT.md"

#: The tables the acceptance test asserts on, next to the cache.
TABLES_NAME = "tables.json"


# ---------------------------------------------------------------------------
# Measuring: one pass over the cache
# ---------------------------------------------------------------------------


def measure(source: Iterable[Entry], per_circuit: int = replay.PER_CIRCUIT,
            draws: int = tightness.DRAWS) -> dict[str, Any]:
    """Every table of `acceptance.md` §2, from one pass over the cache."""
    from testing.phase0.corpus import parity_set

    circuits = {c.label: c for c in parity_set()}
    tables: dict[str, list[Any]] = {
        "invariants": [], "sizes": [], "capacities": [], "questions": [],
        "tightness": [], "selection": [], "replay": [],
    }
    for entry in source:
        tables["invariants"].append(invariants.rows(entry))
        for name, module in (("sizes", sizes), ("questions", questions), ("selection", selection)):
            row = module.rows(entry)
            if row is not None:
                tables[name].append(row)
        tables["capacities"].extend(capacities.rows(entry))
        tables["replay"].extend(replay.rows(entry, per_circuit))
        circuit = circuits.get(entry.label)
        if circuit is not None:
            tables["tightness"].extend(tightness.rows(entry, circuit, draws))
    summaries = {
        "invariants": invariants.summary(tables["invariants"]),
        "sizes": sizes.summary(tables["sizes"]),
        "capacities": capacities.summary(tables["capacities"]),
        "questions": questions.summary(tables["questions"]),
        "tightness": tightness.summary(tables["tightness"]),
        "selection": selection.summary(tables["selection"]),
        "replay": replay.summary(tables["replay"]),
    }
    # The per-answer lists are large and only their statistics are read.
    for row in tables["questions"]:
        row.pop("cones", None)
        row.pop("query_millis", None)
    return {"tables": tables, "summaries": summaries,
            "settings": {"per_circuit": per_circuit, "draws": draws}}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Counter):
        return dict(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and value in (float("inf"), float("-inf")):
        return None
    return value


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _pct(share: float) -> str:
    return f"{share * 100:.0f}%"


def _factor(value: float | None) -> str:
    if value is None:
        return "-"
    if value >= 1e6:
        return f"{value:.1e}"
    return f"{value:.3f}"


def _families(rows: list[dict[str, Any]]) -> list[str]:
    return sorted({r["family"] for r in rows})


def section_invariants(inv: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    families = _families(rows)
    per_family = []
    for family in [*families, None]:
        subset = [r for r in rows if family is None or r["family"] == family]
        per_family.append([
            "**total**" if family is None else family,
            len(subset),
            sum(1 for r in subset if r["status"] == "phase0_rejected"),
            sum(1 for r in subset if r["status"] != "phase0_rejected"),
            sum(1 for r in subset if r["status"] == "ok"),
            sum(1 for r in subset if r["status"] == "phase1_rejected"),
            sum(1 for r in subset if r["status"] not in ("ok", "phase1_rejected", "phase0_rejected")),
        ])
    categories = [[category, inv["categories"].get(category, 0)]
                  for category in invariants.CATEGORIES]
    rejected = sorted((r for r in rows if r["status"] == "phase1_rejected"),
                      key=lambda r: (invariants.CATEGORIES.index(r["category"]), r["label"]))
    listing = [[r["label"], r["category"], r["line"]] for r in rejected]
    failures = [[r["label"], r["status"], r["detail"]] for r in rows
                if r["status"] not in ("ok", "phase1_rejected", "phase0_rejected")]
    unnamed = inv["unnamed_answers"]
    reading = (
        f"Phase 0 accepts {inv['seen_by_phase1']} of the {inv['parity']} parity circuits; "
        f"phase 1 accepts {inv['accepted']} of those {inv['seen_by_phase1']} and rejects "
        f"{inv['rejected']}, with {inv['failed']} harness failures (a panic, a timeout or the "
        f"memory cap). Every rejection carries a reason: {inv['categories'].get('rule gap', 0)} "
        f"name a rule of the library that has no instance over the interface the circuit needs, "
        f"{inv['categories'].get('phase 0 record gap', 0)} are relations Φ does not carry "
        f"(`rules/README.md` §10), {inv['categories'].get('outside the library by decision', 0)} "
        f"are operations `rules/README.md` §9 leaves out, and "
        f"{inv['categories'].get('design gap', 0)} are the design's own limit: a child read "
        f"under a condition that does not imply its ψ. The categories are assigned by the "
        f"patterns of `invariants.category_of`, and the six circuits the message alone does "
        f"not place are decided in `invariants.DECIDED` by reading the kernel. "
        f"On the accepted circuits the seven invariants of `README.md` §2 are re-checked on "
        f"the exported JSON, independently of `src/phase1/graph.rs`: "
        f"{inv['accepted'] - len(inv['with_violations'])} of {inv['accepted']} pass all of them"
        + (f"; the violations are on {', '.join(inv['with_violations'])}" if inv["with_violations"] else "")
        + f". The Rust checker agrees on every circuit"
        + (f" except {', '.join(inv['with_rust_violations'])}" if inv["with_rust_violations"] else "")
        + f". One direction of \"A holds exactly the questions the nodes asked\" cannot be "
        f"re-checked from the export, which does not carry the list of answers the construction "
        f"asked outside a node: {sum(unnamed.values())} answers are named by no node, of which "
        f"{unnamed.get('CANFAIL', 0)} are the CANFAIL sweep's and "
        f"{sum(v for k, v in unnamed.items() if k != 'CANFAIL')} are premises of rule instances "
        f"that produced no node."
    )
    return "\n".join([
        "Accepted and rejected per family, against phase 0's count (`acceptance.md` §2.1).\n",
        table(["family", "parity", "phase 0 rejected", "seen by phase 1", "accepted",
               "rejected by phase 1", "harness failures"], per_family),
        "\nRejections by category.\n",
        table(["category", "circuits"], categories),
        "\nEvery circuit phase 1 rejects, with what is missing.\n",
        table(["circuit", "category", "what is missing"], listing),
        "\nHarness failures: a circuit that panicked, timed out or exceeded the memory cap.\n",
        table(["circuit", "status", "detail"], failures),
        "\nInvariant violations found on the exported JSON, by invariant.\n",
        histogram(Counter(inv["violation_kinds"])),
        "",
        reading,
        "",
    ])


def section_sizes(sz: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    def cell(values: list[float], integer: bool = True) -> str:
        if not values:
            return "-"
        mid, top = median(values), max(values)
        return f"{mid:.0f} / {top:.0f}" if integer else f"{mid:.2f} / {top:.2f}"
    keys = ("delta", "phi", "checks", "requirements", "implementations", "pruned", "rows",
            "terms_built")
    per_family = []
    for family in [*_families(rows), None]:
        subset = [r for r in rows if family is None or r["family"] == family]
        per_family.append([
            "**total**" if family is None else family, len(subset),
            *[cell([r[k] for r in subset]) for k in keys],
            cell([r["ratio"] for r in subset], integer=False),
        ])
    largest = sorted(rows, key=lambda r: -r["nodes"])
    detail, note = capped([
        [r["label"], r["delta"], r["phi"], r["checks"], r["requirements"], r["implementations"],
         r["pruned"], r["rows"], r["terms_built"], f"{r['ratio']:.2f}"]
        for r in largest
    ])
    ratio = sz["ratio"]
    reading = (
        f"Over the {sz['circuits']} accepted circuits G holds {sz['requirements']} requirement "
        f"and {sz['implementations']} implementation nodes against |Δ| = {sz['delta']} entries "
        f"in total; the dominance pruning of `README.md` §4 dropped {sz['pruned']} further "
        f"nodes, and phase 1 built {sz['terms_built']} terms over Φ's variables. The ratio of "
        f"nodes to |Δ| is {ratio['median']:.2f} at the median and ranges from {ratio['min']:.2f} "
        f"to {ratio['max']:.2f}"
        + (f"; it exceeds 10 on {len(sz['ratio_over_10'])} circuits "
           f"({', '.join(sz['ratio_over_10'][:6])}{', …' if len(sz['ratio_over_10']) > 6 else ''})"
           if sz["ratio_over_10"] else "")
        + f". The largest graphs are {', '.join(sz['largest'][:3])}. The highest ratios are "
        + "; ".join(f"{label} at {ratio:.1f}, whose nodes are mostly "
                    f"{', '.join(rules)}" for label, ratio, rules in sz["highest_ratio"])
        + "; the lowest are "
        + "; ".join(f"{label} at {ratio:.2f} ({', '.join(rules)})"
                    for label, ratio, rules in sz["lowest_ratio"])
        + ". A ratio bounded by a small constant across three orders of magnitude of |Δ| is "
        f"what §11's first row asks for: the graph grows linearly in the record, with the "
        f"constant set by how many interfaces a value is offered in and by the reading rules, "
        f"which instantiate one node per provider representation of the value they read."
    )
    return "\n".join([
        "Graph size against program size per family; each cell is median / maximum over the "
        "accepted circuits (`acceptance.md` §2.2).\n",
        table(["family", "circuits", "|Δ|", "|Φ|", "checks", "requirements",
               "implementations", "pruned", "rows", "terms built", "nodes / |Δ|"], per_family),
        "\nThe largest graphs by node count.\n",
        table(["circuit", "|Δ|", "|Φ|", "checks", "req", "impl", "pruned", "rows",
               "terms", "nodes / |Δ|"], detail) + note,
        "",
        reading,
        "",
    ])


def section_capacities(cap: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    def show(value: Any) -> str:
        return "-" if value is None else str(value)
    ordered = sorted(rows, key=lambda r: (not r["below_trivial"], not r["below_declared"],
                                          r["label"], r["answer"]))
    detail, note = capped([
        [r["label"], ", ".join(r["rules"]), r["term"], show(r["trivial"]), show(r["derived"]),
         show(r["declared"]) + (f" ({r['declared_by']})" if r["declared_by"] else ""),
         r["layer"], "yes" if r["tight"] else "no"]
        for r in ordered
    ])
    factors = cap["factors"]
    small = [f for f in factors if f < 1.01]
    large = [f for f in factors if f >= 1e6]
    middle = [f for f in factors if 1.01 <= f < 1e6]
    reading = (
        f"{cap['capacities']} capacities were decided by a MAX over {cap['circuits']} circuits "
        f"(one row per distinct MAX answer an implementation node still in G names; "
        f"{cap['no_bound']} answered with no bound). {cap['equal_trivial']} equal the trivial "
        f"bound and {cap['below_trivial']} are strictly below it"
        + (f", on {', '.join(cap['below_trivial_labels'])}, by factors of "
           f"{', '.join(_factor(f) for f in factors[:10])}" if factors else "")
        + f". Against the declared bound — the annotation bound of the input the term depends "
        f"on, read from the `var_range` of its shape symbols and input lengths and narrowed by a "
        f"`precondition` where one exists — {cap['below_declared']} are below it and "
        f"{cap['equal_declared']} equal to it; {cap['no_declared']} depend on no declared input "
        f"at all, because the corpus's DS-1000 kernels fix their shapes in the annotation and the "
        f"capacity is then a constant of the record. No parity circuit carries a `precondition` "
        f"constraint, so the declared column is the annotation alone. The interval layer answered "
        f"{cap['by_layer']['interval']} of these MAX questions and Z3 {cap['by_layer']['z3']}; "
        f"{cap['tight']} carry the oracle's tightness flag. What the corpus shows is that the "
        f"derived bound is rarely below the trivial one: "
        f"{len(small)} of the reductions are by a factor under 1.01 (a one-off `N − 1` of a "
        f"slice or a difference), {len(middle)} by a factor between 1.01 and 10^6, and "
        f"{len(large)} by a factor above 10^6 (a dtype-range capacity brought down to a "
        f"declared size by the `size_product` declarations of `rules/README.md` §10 item 12). "
        f"No capacity of this corpus is a count bounded by a relation between arrays, which is "
        f"the case the mechanism was built for; §11's second row is not confirmed on this "
        f"corpus and stays open on one that has such programs."
    )
    return "\n".join([
        "Capacities decided by a MAX: the trivial bound (intervals over the ranges alone), the "
        "derived bound (the MAX verdict) and the declared bound (the annotation of the input the "
        "term depends on), with the layer that answered (`acceptance.md` §2.3). Rows below the "
        "trivial bound first.\n",
        table(["circuit", "rules", "term", "trivial", "derived", "declared", "layer", "tight"],
              detail) + note,
        "",
        reading,
        "",
    ])


def section_questions(qs: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    kinds = [[kind, qs["by_kind"][kind], *[qs["kind_layers"][kind][layer] for layer in questions.LAYERS]]
             for kind in questions.KINDS]
    kinds.append(["**total**", qs["questions"], *[qs["by_layer"][layer] for layer in questions.LAYERS]])
    solver = [
        ["Z3 queries (sliced path and sweep)", qs["queries"]],
        ["sat", qs["sat"]], ["unsat", qs["unsat"]], ["unknown", qs["unknown"]],
        ["answers recorded as timed out", qs["unknown_answers"]],
        ["CONST answered not constant because the model gave the term no value",
         f"{qs['const_without_value']} on {', '.join(qs['const_without_value_labels']) or 'none'}"],
        ["memo hit rate", _pct(qs["memo_rate"])],
        ["questions answered without Z3", _pct(qs["without_solver"])],
        ["cone constraints, sliced path: mean / p95 / max",
         f"{qs['cone_mean']:.1f} / {qs['cone_p95']} / {qs['cone_max']}"],
        ["query milliseconds: mean / p95 / max",
         f"{qs['query_mean']:.2f} / {qs['query_p95']} / {qs['query_max']}"],
        ["solver time, total ms", qs["millis"]],
        ["CANFAIL sweep: queries / constraints asserted / ms",
         f"{qs['sweep_queries']} / {qs['sweep_asserted']} / {qs['sweep_millis']}"],
        ["constraints of Φ over the accepted circuits", qs["phi_constraints"]],
    ]
    ordered = sorted(rows, key=lambda r: (-r["millis"], -r["questions"], r["label"]))
    detail, note = capped([
        [r["label"], r["questions"], _pct(r["memo_rate"]), r["queries"], r["unsat"], r["unknown"],
         f"{r['cone_mean']:.1f}", r["cone_max"], f"{r['query_mean']:.1f}", r["query_p95"],
         r["query_max"], r["millis"]]
        for r in ordered
    ])
    slow = [[label, millis, queries, cone, _slow_reason(label, rows)]
            for label, millis, queries, cone in qs["slow"]]
    reading = (
        f"The {qs['circuits']} accepted circuits asked {qs['questions']} questions, "
        f"{_pct(qs['memo_rate'])} of them answered by the memo and "
        f"{_pct(qs['without_solver'])} without Z3 at all; the {qs['queries']} Z3 queries "
        f"came back {qs['sat']} sat, {qs['unsat']} unsat and {qs['unknown']} unknown, in "
        f"{qs['millis']} ms in total. On the sliced path a query asserts "
        f"{qs['cone_mean']:.1f} constraints on average and {qs['cone_max']} at most, which is "
        f"the dependency cone of `oracle/mod.rs` and not Φ (the accepted circuits hold "
        f"{qs['phi_constraints']} constraints between them); a query takes "
        f"{qs['query_mean']:.2f} ms on average, {qs['query_p95']} ms at the 95th percentile and "
        f"{qs['query_max']} ms at most. The CANFAIL sweep asked {qs['sweep_queries']} questions "
        f"over one incremental solver that asserted {qs['sweep_asserted']} constraints in total. "
        + (f"{len(qs['slow'])} circuits stand above {questions.SLOW_MILLIS} ms "
           f"({', '.join(label for label, *_ in qs['slow'])}): each has thousands of checks, "
           f"each check is one CANFAIL question, and the table above says how much of the time "
           f"is the sweep's — one query per check on a solver that holds every earlier check — "
           f"and how much the sliced CONST questions of the reads took. "
           if qs["slow"] else "No circuit is above 5 s of solver time. ")
        + f"The millisecond answers of §11's third row hold per question; the total is linear "
        f"in the number of checks Φ holds, and a loop unrolled n times contributes n of them. "
        + (f"{qs['const_without_value']} CONST questions were answered \"not constant\" because "
           f"Z3's model gave the term no value (`oracle/mod.rs`): the answer is the conservative "
           f"one §5.5 prescribes for an unknown, and it costs the read a `READ_DIRECT` "
           f"candidate it might have had."
           if qs["const_without_value"] else "")
    )
    return "\n".join([
        "Questions by kind and by answering layer over the accepted circuits "
        "(`acceptance.md` §2.4).\n",
        table(["kind", "asked", *questions.LAYERS], kinds),
        "\nThe solver, in total.\n",
        table(["measure", "value"], solver),
        "\nPer circuit, by solver time; cone sizes are over the sliced path, query times over "
        "every Z3 answer.\n",
        table(["circuit", "Q", "memo", "z3", "unsat", "unknown", "cone mean", "cone max",
               "ms mean", "ms p95", "ms max", "ms total"], detail) + note,
        f"\nCircuits over {questions.SLOW_MILLIS} ms of solver time, with where the time went.\n",
        table(["circuit", "ms", "queries", "largest cone", "where the time went"], slow),
        "",
        reading,
        "",
    ])


def _slow_reason(label: str, rows: list[dict[str, Any]]) -> str:
    row = next(r for r in rows if r["label"] == label)
    sweep_share = row["sweep_millis"] / row["millis"] if row["millis"] else 0.0
    return (f"{row['sweep_queries']} CANFAIL sweep queries took {row['sweep_millis']} ms "
            f"({_pct(sweep_share)}); the other {row['queries'] - row['sweep_queries']} queries "
            f"the rest, at {row['query_mean']:.1f} ms each")


def section_tightness(tg: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    verdicts = [[verdict, count] for verdict, count in sorted(tg["verdicts"].items())]
    interesting = [r for r in rows if r["verdict"] != "tight"]
    detail, note = capped([
        [r["label"], r["term"], r["bound"], r["reached"], r["runs"], r["verdict"],
         _tightness_note(r)]
        for r in sorted(interesting, key=lambda r: (r["label"], r["answer"] or 0))
    ])
    unrealised = tg["unrealised_kinds"]
    reading = (
        f"{tg['maxes']} MAX answers over {tg['circuits']} circuits were answered by Z3 with a "
        f"model. For each, the harness built inputs from the model — shape symbols, input "
        f"lengths and scalar inputs set to the model's values — ran the source with the "
        f"reference evaluator, and evaluated the MAX's term under θ(u) of the run; when the "
        f"model-shaped run did not reach the bound, {tightness.DRAWS} further draws of the same "
        f"shapes were run, alternately from the benign element domain and from "
        f"[−{tightness.WIDE}, {tightness.WIDE}]. {tg['tight']} bounds were reached "
        f"(tight), {tg['loose']} were not reached by any run although the model pins nothing "
        f"the generator cannot set (loose), {tg['unrealised']} were not reached and the model "
        f"pins a variable the generator cannot target (unrealised), {tg['unsound']} were "
        f"exceeded by a run (a defect), and {tg['not_run']} could not be run. What the "
        f"generator cannot realise is a value the program computes: a count of a mask, a "
        f"`unique` count, a value-sized length, a data scalar that is not an input"
        + (f" — here {', '.join(f'`{k}`' for k in unrealised)}" if unrealised else "")
        + ". Where such a variable took one and the same value on every run while the model "
        "pins it elsewhere, Φ lacks the relation that fixes it, and the row says so."
    )
    return "\n".join([
        "Tightness of the MAX answers Z3 gave with a model (`acceptance.md` §2.5).\n",
        table(["verdict", "MAX answers"], verdicts),
        "\nEvery MAX not reached by a run, with what the model pins and what the runs gave.\n",
        table(["circuit", "term", "bound", "reached", "runs", "verdict", "note"], detail) + note,
        "",
        reading,
        "",
    ])


def _tightness_note(row: dict[str, Any]) -> str:
    if row["note"]:
        return row["note"]
    parts = []
    for name in row.get("unsettable", []):
        pinned = row["model"].get(name)
        observed = row.get("observed", {}).get(name, [])
        seen = ", ".join(str(v) for v in observed[:6]) or "no value"
        parts.append(f"the model pins `{name}` to {pinned}; the runs gave {seen}")
        if len(observed) == 1 and pinned is not None and observed[0] != pinned:
            parts.append(f"Φ lacks the relation that fixes `{name}` at {observed[0]}")
    return "; ".join(parts)


def section_selection(sel: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    names = [*selection.VARIANTS, "force_packed_literal"]
    words = {**{name: w for name, (w, _) in selection.VARIANTS.items()},
             "force_packed_literal": "BUILD_MASKED (the literal reading)"}
    base = sel["greedy_comparable"]
    variants = [["greedy, unrestricted", base, "-", "-", "-", "-"]]
    for name in names:
        variants.append([
            f"{name} (drop {words[name]})", sel[name],
            f"{sel[name] / base:.3f}" if base else "-",
            sel[f"{name}_dearer"], sel[f"{name}_cheaper"],
            len(sel[f"{name}_unrealisable"]),
        ])
    ordered = sorted(rows, key=lambda r: (-r["greedy"], r["label"]))

    def cell(row: dict[str, Any], name: str) -> str:
        unfilled = len(row[f"{name}_unfilled"])
        if row[name] is None:
            return f"unrealisable ({unfilled})"
        return f"{row[name]} ({unfilled})" if unfilled else str(row[name])

    detail, note = capped([
        [r["label"], r["requirements"], r["nodes"], r["greedy"],
         *[cell(r, name) for name in selection.VARIANTS]]
        for r in ordered
    ])
    unrealisable = [[name, ", ".join(sel[f"{name}_unrealisable"]),
                     "; ".join(f"{what} × {count}" for what, count in
                               sel[f"{name}_unfilled"].most_common(4))]
                    for name in names if sel[f"{name}_unrealisable"]]
    rules = [[rule, count, *[sel[f"{name}_selected_rules"].get(rule, 0)
                             for name in selection.VARIANTS]]
             for rule, count in sel["selected_rules"].most_common()]
    cheaper = sorted({r["label"] for r in rows
                      if any(r[name] is not None and r[name] < r["greedy"]
                             for name in selection.VARIANTS)})
    reading = (
        f"The greedy selection over the {sel['circuits']} accepted graphs chooses "
        f"{sel['requirements']} requirements' providers and selects {sel['nodes']} distinct "
        f"implementation nodes at a total of {sel['greedy']} units. On the {sel['comparable']} "
        f"circuits every variant is realisable on, the unrestricted greedy costs {base} units; "
        f"forcing every read with a ROM alternative through the ROM costs "
        f"{sel['force_rom'] / base:.3f} of that, forcing the key mux {sel['force_mux'] / base:.3f}, "
        f"and forcing the consumers of a masked bundle onto the packed forms "
        f"{sel['force_packed'] / base:.3f}; a forced variant is dearer on "
        f"{sel['force_rom_dearer']}, {sel['force_mux_dearer']} and "
        f"{sel['force_packed_dearer']} circuits respectively. The two reading variants filter "
        f"only where an alternative exists — a read whose coordinates are all constant has "
        f"`READ_DIRECT` alone — so they are realisable everywhere; the packed variant leaves "
        f"a requirement without a provider on some circuits (the number in parentheses) and "
        f"the choice then moves up to a parent with another subtree, which is what makes it "
        f"realisable on {sel['force_packed_realisable']} circuits. The literal reading, "
        f"dropping `BUILD_MASKED` itself, is unrealisable on "
        f"{len(sel['force_packed_literal_unrealisable'])} circuits and identical to the "
        f"unrestricted greedy on the rest, because `PACK_BY_KEY` builds the packed form out of "
        f"the masked bundle and the masked bundle is the only provider of its requirement. "
        + (f"On {len(cheaper)} circuits ({', '.join(cheaper[:6])}) a forced variant is *cheaper* "
           f"than the unrestricted greedy: the greedy chooses each provider on an additive "
           f"estimate that counts a shared child once per parent, and the realised cost counts "
           f"it once, so the choice is not optimal on graphs with shared nodes. That gap is what "
           f"phase 2's global selection is for (§11, row 6), and on this corpus it is at least "
           f"the difference shown. "
           if cheaper else "")
        + "Every cost is a `UnitCost` (`src/phase1/cost.rs`): the primitive counts of "
        "`rules/README.md` §7 at weight 1, `perm`, `lookup`, `hash` and `cell` at the "
        "placeholder 0 and every `fop` at 1. They are counts of primitives, not gates, and a "
        "backend cost function replaces them without touching a rule."
    )
    return "\n".join([
        "The greedy selection and the forced variants of `revision/README.md` §11, in "
        "`UnitCost` units over the circuits on which every variant is realisable "
        "(`acceptance.md` §2.6).\n",
        table(["selection", "total units", "of greedy", "dearer on", "cheaper on",
               "unrealisable on"], variants),
        "\nPer circuit, by greedy cost; a forced variant's cell is its cost, with the number of "
        "requirements the filter left without a provider in parentheses where there are any.\n",
        table(["circuit", "req", "nodes", "greedy", *selection.VARIANTS], detail) + note,
        "\nWhere a forced variant leaves a root without a node.\n",
        table(["variant", "circuits", "requirements left without a provider"], unrealisable),
        "\nRules selected, by how many requirements chose them, under the greedy and under "
        "each forced variant.\n",
        table(["rule", "greedy", *selection.VARIANTS], rules),
        "",
        reading,
        "",
    ])


def section_replay(rp: dict[str, Any], rows: list[dict[str, Any]], per_circuit: int) -> str:
    by_kind = [[kind, rp["by_kind"][kind]["interval"], rp["by_kind"][kind]["z3"]]
               for kind in replay.KINDS]
    outcomes = [[name, rp[name]] for name in ("agree", "disagree", "weaker", "unknown",
                                               "skipped", "unreplayable")]
    problems = [[r["label"], r["answer"], r["layer"], r["kind"], r["verdict"], r["detail"]]
                for r in [*rp["disagreements"], *rp["weaker_rows"]]]
    reading = (
        f"{rp['replayed']} answers were replayed over {rp['circuits']} circuits: every "
        f"answer of the interval layer ({rp['interval']}), by interval arithmetic over the "
        f"exported ranges, and up to {per_circuit} answers of the Z3 layer per circuit "
        f"({rp['z3']}), spread over the kinds, by `phi_solver.py`'s arena over the constraints "
        f"of Φ_e under the answer's condition. {rp['agree']} agree, {rp['disagree']} disagree, "
        f"{rp['weaker']} came back unsatisfiable over the whole of Φ_e where the oracle's cone "
        f"had a model, {rp['unknown']} timed out on the replay and {rp['skipped']} were "
        f"skipped for the size of the arena. A disagreement is a defect and is listed above "
        f"with the circuit, the question and both verdicts"
        + ("; there is none." if not rp["disagree"] else ".")
        + " The replay asserts the whole of Φ_e rather than the oracle's cone, which the export "
        "does not carry; that makes the unsatisfiable direction a strict check and the "
        "satisfiable one a check on Φ_e's other components. MAX and MIN answers are replayed "
        "as well, as the bound's soundness (`e > b` unsatisfiable), beyond the three kinds "
        "`acceptance.md` §2.7 names."
    )
    return "\n".join([
        "Answers replayed, by kind and layer (`acceptance.md` §2.7).\n",
        table(["kind", "interval layer", "Z3 layer"], by_kind),
        "\nOutcomes.\n",
        table(["outcome", "answers"], outcomes),
        "\nEvery disagreement and every answer weaker over the whole of Φ_e.\n",
        table(["circuit", "answer", "layer", "kind", "verdict", "detail"], problems),
        "",
        reading,
        "",
    ])


def section_criteria(measured: dict[str, Any]) -> str:
    s = measured["summaries"]
    inv, rp = s["invariants"], s["replay"]
    criteria = [
        ["every accepted circuit passes every invariant of §2.1",
         f"{inv['accepted'] - len(inv['with_violations'])} of {inv['accepted']}",
         "pass" if not inv["with_violations"] else "FAIL"],
        ["no circuit panics or times out; every rejection carries a reason",
         f"{inv['failed']} harness failures, {inv['rejected']} rejections with a reason",
         "pass" if not inv["failed"] else "FAIL"],
        ["no replayed answer disagrees",
         f"{rp['disagree']} of {rp['replayed']}",
         "pass" if not rp["disagree"] else "FAIL"],
    ]
    reported = [
        ["acceptance against phase 0's 150", f"{inv['accepted']} accepted, {inv['rejected']} rejected"],
        ["graph-size ratio nodes / |Δ|, median (min – max)",
         f"{s['sizes']['ratio']['median']:.2f} ({s['sizes']['ratio']['min']:.2f} – "
         f"{s['sizes']['ratio']['max']:.2f})"],
        ["capacities below the trivial bound",
         f"{s['capacities']['below_trivial']} of {s['capacities']['capacities']}"],
        ["questions / Z3 queries / solver ms",
         f"{s['questions']['questions']} / {s['questions']['queries']} / {s['questions']['millis']}"],
        ["tightness: tight / loose / unrealised / unsound",
         f"{s['tightness']['tight']} / {s['tightness']['loose']} / "
         f"{s['tightness']['unrealised']} / {s['tightness']['unsound']}"],
        ["greedy / force_rom / force_mux / force_packed, units, on the comparable circuits",
         f"{s['selection']['greedy_comparable']} / {s['selection']['force_rom']} / "
         f"{s['selection']['force_mux']} / {s['selection']['force_packed']}"],
    ]
    return "\n".join([
        "The three criteria `test_acceptance.py` asserts (`acceptance.md` §4).\n",
        table(["criterion", "measured", "verdict"], criteria),
        "\nReport-only numbers, stated and not asserted.\n",
        table(["measure", "value"], reported),
        "",
    ])


def render(measured: dict[str, Any], cache: Path) -> str:
    s, t = measured["summaries"], measured["tables"]
    absent = missing(cache)
    head = (
        "Generated by `testing/phase1/report.py` from the per-circuit cache "
        f"(`{cache}`, {size_bytes(cache) / 1e6:.1f} MB, one JSON per parity circuit with the "
        "phase 0 record, the phase 1 record and the timings; "
        f"{len(absent)} parity circuits have no cache file). The sections are those of "
        "`revision/phase1/acceptance.md` §2, one measurement each, with the acceptance criteria "
        "of §4 last."
    )
    return "\n".join([
        "# Phase 1 acceptance report", "", head, "",
        "## 1. Acceptance and invariants", "",
        section_invariants(s["invariants"], t["invariants"]),
        "## 2. Graph size against program size", "",
        section_sizes(s["sizes"], t["sizes"]),
        "## 3. Capacities, three columns", "",
        section_capacities(s["capacities"], t["capacities"]),
        "## 4. Questions, memo, solver time, unknowns", "",
        section_questions(s["questions"], t["questions"]),
        "## 5. Tightness verification", "",
        section_tightness(s["tightness"], t["tightness"]),
        "## 6. Baseline path and greedy cost", "",
        section_selection(s["selection"], t["selection"]),
        "## 7. Evidence replay", "",
        section_replay(s["replay"], t["replay"], measured["settings"]["per_circuit"]),
        "## 8. Acceptance criteria", "",
        section_criteria(measured),
    ])


def tables_path(cache: Path = CACHE_DIR) -> Path:
    return cache / TABLES_NAME


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, default=CACHE_DIR)
    parser.add_argument("--per-circuit", type=int, default=replay.PER_CIRCUIT,
                        help="Z3 answers replayed per circuit")
    parser.add_argument("--draws", type=int, default=tightness.DRAWS,
                        help="random draws per MAX in the tightness check")
    arguments = parser.parse_args()
    measured = measure(entries(arguments.cache), arguments.per_circuit, arguments.draws)
    tables_path(arguments.cache).write_text(
        json.dumps(_jsonable(measured), indent=1, sort_keys=True), encoding="utf-8")
    RESULTS.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text(render(measured, arguments.cache), encoding="utf-8")
    print(f"wrote {REPORT_MD} and {tables_path(arguments.cache)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
