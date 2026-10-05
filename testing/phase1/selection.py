"""Baseline path and greedy cost (`revision/phase1/acceptance.md` §2.6;
`revision/README.md` §11, rows 6 and 7, the part phase 1 can answer before
phase 2 exists).

A greedy selection over the exported G: every requirement reachable from a
root takes its cheapest implementation node, where the cost of a node is
its own cost plus the estimated cost of its child requirements, computed in
one pass in reverse topological order (G is acyclic). The estimate counts a
shared child once per parent; the *realised* cost of the selection then
counts every selected node once — the selected subgraph is walked from the
roots and its distinct implementation nodes are summed — which is the
number reported.

Three forced variants of §11 are the same pass over a filtered graph:

| variant | filter on a requirement's providers |
|---|---|
| `force_rom` | where a `READ_ROM` provider exists, drop `READ_KEY_MUX` and `READ_DIRECT` |
| `force_mux` | where a `READ_KEY_MUX` provider exists, drop `READ_ROM` and `READ_DIRECT` |
| `force_packed` | drop every node that reads a `Masked` bundle, except the rules that build the packed and counted forms out of it (:data:`PACKERS`) |

The two reading variants filter "where an alternative exists": a read whose
every coordinate is constant has `READ_DIRECT` alone (`README.md` §4 step
4) and cannot be forced. `force_packed` is the consumer-side reading of
"`PACK_BY_KEY` over `BUILD_MASKED`": the packed form of a filter is built
*from* the masked one (`PACK_BY_KEY` reads the `Masked` bundle), so dropping
`BUILD_MASKED` itself drops the packed form with it — that number is still
computed, as `force_packed_literal`, and reported for the record. A
requirement the filter leaves without a provider has no cost, and neither
has any node that needs it, so the choice moves up: a parent takes another
subtree if the graph offers one. When a root itself is left without a node,
the variant is *unrealisable* on that circuit; the row names the
requirements left without a provider, and the totals are taken over the
circuits on which every variant is realisable, beside the unrestricted total
over the same circuits, so that the ratios compare like with like.

Every cost is a `UnitCost` (`src/phase1/cost.rs`): the primitive counts of
`rules/README.md` §7 at weight 1, with the backend primitives (`perm`,
`lookup`, `hash`, `cell`) at their placeholder 0 and every `fop` at 1. They
are not gates.

```text
.venv/bin/python -m testing.phase1.selection
```
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from testing.phase1.cache import CACHE_DIR, Entry, entries

#: The rules that read a `Masked` bundle in order to build its packed or
#: counted form, which the forced-packed variant keeps.
PACKERS = ("PACK_BY_KEY", "PREFIX_RANK", "COUNT", "NONZERO", "BUILD_MASKED", "UNIQUE")

#: A filter: the providers of one requirement it admits, given the graph's
#: implementation and requirement lists.
Admit = Callable[[list[int], list[dict[str, Any]], list[dict[str, Any]]], list[int]]


def _prefer(wanted: str, dropped: tuple[str, ...]) -> Admit:
    """Where a `wanted` provider exists, drop the `dropped` rules."""

    def admit(providers: list[int], implementations: list[dict[str, Any]],
              requirements: list[dict[str, Any]]) -> list[int]:
        del requirements
        if not any(implementations[p]["rule"] == wanted for p in providers):
            return providers
        return [p for p in providers if implementations[p]["rule"] not in dropped]

    return admit


def _drop(dropped: tuple[str, ...]) -> Admit:
    return lambda providers, implementations, requirements: [
        p for p in providers if implementations[p]["rule"] not in dropped]


def _off_masked(providers: list[int], implementations: list[dict[str, Any]],
                requirements: list[dict[str, Any]]) -> list[int]:
    def keeps(node: dict[str, Any]) -> bool:
        if node["rule"] in PACKERS:
            return True
        return not any(requirements[at]["family"] == "Masked" for at in node["inputs"])
    return [p for p in providers if keeps(implementations[p])]


#: The forced variants: name → (the filter in words, the filter).
VARIANTS: dict[str, tuple[str, Admit]] = {
    "force_rom": ("READ_KEY_MUX, READ_DIRECT where READ_ROM exists",
                  _prefer("READ_ROM", ("READ_KEY_MUX", "READ_DIRECT"))),
    "force_mux": ("READ_ROM, READ_DIRECT where READ_KEY_MUX exists",
                  _prefer("READ_KEY_MUX", ("READ_ROM", "READ_DIRECT"))),
    "force_packed": ("readers of a Masked bundle", _off_masked),
}

#: The literal reading of the third variant, reported but not tabulated.
FORCE_PACKED_LITERAL: Admit = _drop(("BUILD_MASKED",))

INFINITE = float("inf")


@dataclass
class Selection:
    """What one greedy pass chose."""

    #: Requirement index → the implementation node chosen for it.
    chosen: dict[int, int]
    #: The distinct implementation nodes of the selected subgraph.
    selected: set[int]
    #: Sum of the costs of `selected`, each node once.
    realised: int
    #: The additive estimate summed over the roots, shared nodes counted
    #: once per parent.
    estimate: float
    #: Requirements reachable from a root that the filter left without a
    #: provider, as "form target" strings.
    unfilled: list[str] = field(default_factory=list)
    #: Selected rule → how many requirements it was chosen for.
    rules: Counter = field(default_factory=Counter)

    @property
    def realisable(self) -> bool:
        return self.estimate != INFINITE


def topological(graph: dict[str, Any]) -> list[int]:
    """The requirements reachable from the roots, children before parents."""
    requirements, implementations = graph["requirements"], graph["implementations"]
    order: list[int] = []
    state: dict[int, int] = {}
    for root in graph["roots"]:
        if root in state:
            continue
        stack = [(root, 0)]
        state[root] = 1
        while stack:
            at, cursor = stack[-1]
            children = [c for p in requirements[at]["providers"]
                        for c in implementations[p]["inputs"]]
            if cursor >= len(children):
                state[at] = 2
                order.append(at)
                stack.pop()
                continue
            stack[-1] = (at, cursor + 1)
            child = children[cursor]
            if child not in state:
                state[child] = 1
                stack.append((child, 0))
    return order


def greedy(graph: dict[str, Any], admit: Admit | None = None) -> Selection:
    """One greedy pass; `admit` is the filter of a forced variant (every
    provider when `None`)."""
    requirements, implementations = graph["requirements"], graph["implementations"]
    order = topological(graph)
    estimate: dict[int, float] = {}
    chosen: dict[int, int] = {}
    unfilled: list[str] = []
    for at in order:
        providers = requirements[at]["providers"]
        admitted = providers if admit is None else admit(providers, implementations, requirements)
        if not admitted and providers:
            requirement = requirements[at]
            unfilled.append(f"{requirement['form']} {requirement['target']} "
                            f"{requirement['interface'] or ''}".strip())
        best, best_cost = None, INFINITE
        for provider in admitted:
            node = implementations[provider]
            cost = int(node["cost"]) + sum(estimate.get(child, INFINITE) for child in node["inputs"])
            if cost < best_cost:
                best, best_cost = provider, cost
        if best is not None:
            chosen[at] = best
        estimate[at] = best_cost
    total = sum(estimate.get(root, INFINITE) for root in graph["roots"])
    selected: set[int] = set()
    if total != INFINITE:
        stack = list(graph["roots"])
        seen: set[int] = set()
        while stack:
            at = stack.pop()
            if at in seen:
                continue
            seen.add(at)
            node = implementations[chosen[at]]
            selected.add(chosen[at])
            stack.extend(node["inputs"])
    rules = Counter(implementations[p]["rule"] for p in selected)
    return Selection(
        chosen=chosen,
        selected=selected,
        realised=sum(int(implementations[p]["cost"]) for p in selected),
        estimate=total,
        unfilled=unfilled,
        rules=rules,
    )


def rows(entry: Entry) -> dict[str, Any] | None:
    if not entry.accepted or entry.phase1 is None:
        return None
    graph = entry.phase1
    free = greedy(graph)
    row: dict[str, Any] = {
        "label": entry.label,
        "family": entry.family,
        "requirements": len(free.chosen),
        "nodes": len(free.selected),
        "greedy": free.realised,
        "estimate": free.estimate,
        "rules": dict(free.rules),
        "rules_present": dict(Counter(n["rule"] for n in graph["implementations"]
                                      if not n["dropped"])),
    }
    for name, (_, admit) in VARIANTS.items():
        forced = greedy(graph, admit)
        row[name] = forced.realised if forced.realisable else None
        row[f"{name}_unfilled"] = forced.unfilled
        row[f"{name}_rules"] = dict(forced.rules)
    literal = greedy(graph, FORCE_PACKED_LITERAL)
    row["force_packed_literal"] = literal.realised if literal.realisable else None
    row["force_packed_literal_unfilled"] = literal.unfilled
    return row


def summary(table: list[dict[str, Any]]) -> dict[str, Any]:
    names = [*VARIANTS, "force_packed_literal"]
    comparable = [r for r in table if all(r[name] is not None for name in names)]
    out: dict[str, Any] = {
        "circuits": len(table),
        "greedy": sum(r["greedy"] for r in table),
        "nodes": sum(r["nodes"] for r in table),
        "requirements": sum(r["requirements"] for r in table),
        "selected_rules": sum((Counter(r["rules"]) for r in table), Counter()),
        "comparable": len(comparable),
        "greedy_comparable": sum(r["greedy"] for r in comparable),
    }
    for name in names:
        realisable = [r for r in table if r[name] is not None]
        out[name] = sum(r[name] for r in comparable)
        out[f"{name}_realisable"] = len(realisable)
        out[f"{name}_unrealisable"] = [r["label"] for r in table if r[name] is None]
        out[f"{name}_dearer"] = sum(1 for r in realisable if r[name] > r["greedy"])
        out[f"{name}_cheaper"] = sum(1 for r in realisable if r[name] < r["greedy"])
        out[f"{name}_selected_rules"] = sum(
            (Counter(r.get(f"{name}_rules", {})) for r in realisable), Counter())
        out[f"{name}_unfilled"] = Counter(
            u.split(" ", 1)[0] + " " + (u.split(" ")[2] if len(u.split(" ")) > 2 else "")
            for r in table if r[name] is None for u in r[f"{name}_unfilled"]
        )
    return out


def table(source: Iterable[Entry]) -> list[dict[str, Any]]:
    return [row for row in (rows(entry) for entry in source) if row is not None]


def main() -> int:
    result = table(entries(CACHE_DIR))
    header = (f"{'circuit':<46}{'req':>6}{'nodes':>6}{'greedy':>12}"
              + "".join(f"{name:>14}{'unf':>5}" for name in VARIANTS))
    print(header)
    print("-" * len(header))
    for row in sorted(result, key=lambda r: -r["greedy"]):
        print(f"{row['label'][:45]:<46}{row['requirements']:>6}{row['nodes']:>6}{row['greedy']:>12}"
              + "".join(f"{'-' if row[name] is None else row[name]:>14}"
                        f"{len(row[name + '_unfilled']):>5}" for name in VARIANTS))
    totals = summary(result)
    print("-" * len(header))
    print(f"{totals['circuits']} circuits, greedy total {totals['greedy']} units over "
          f"{totals['nodes']} selected nodes; on the {totals['comparable']} circuits every "
          f"variant is realisable on, greedy {totals['greedy_comparable']}; "
          + "; ".join(f"{name} {totals[name]} ({totals[name + '_dearer']} dearer, "
                      f"{totals[name + '_cheaper']} cheaper, "
                      f"{len(totals[name + '_unrealisable'])} unrealisable)"
                      for name in [*VARIANTS, "force_packed_literal"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
