"""Capacities, three columns (`revision/phase1/acceptance.md` §2.3;
`revision/README.md` §11, row 2: the derived bound is tighter than the
trivial one).

One row per capacity a MAX decided in an accepted circuit — a distinct MAX
answer some implementation node still in G names — with:

- the *trivial* bound: interval arithmetic over the `var_range` constraints
  alone, which the oracle records as the answer's `interval_bound`;
- the *derived* bound: the MAX verdict over Φ_e ∧ g;
- the *declared* bound: the bound the source declared for the inputs the
  term depends on — the largest `hi` of the `var_range` of an
  `input_length` or `shape_symbol` variable reached from the term, narrowed
  by a `precondition` of the form `x ≤ c` where one exists — and the name
  of the variable that gave it;
- the layer that answered, and the oracle's own tightness flag.

The term of a capacity depends on inputs through Φ's variables: a count `c`
is bounded above by the length of the mask it counts, which is a term over
the input lengths, so the walk from a term to the inputs it depends on goes
through the range bounds of the variables it meets (:func:`input_variables`).

```text
.venv/bin/python -m testing.phase1.capacities
```
"""

from __future__ import annotations

from typing import Any, Iterable

from testing.phase0.terms import TermDag, UnboundVariable
from testing.phase1.cache import CACHE_DIR, Entry, entries

#: The variable origins whose range is what the source declared: an axis
#: bound of an annotation, or a shape symbol's.
DECLARED_ORIGINS = ("input_length", "shape_symbol")


def closed_value(dag: TermDag, node: Any) -> int | None:
    """A node's value when it mentions no variable, else `None`."""
    if node is None:
        return None
    try:
        return int(dag.evaluate(int(node), {}))
    except (UnboundVariable, ValueError, TypeError):
        return None


def precondition_bounds(dag: TermDag, record: dict[str, Any]) -> dict[int, int]:
    """Upper bounds a `precondition` states on a variable outright: every
    conjunct `x ≤ c` or `x < c` of a precondition formula, by variable id."""
    bounds: dict[int, int] = {}
    for constraint in record["constraints"]:
        if constraint["kind"] != "precondition":
            continue
        stack = [int(constraint["formula"])]
        while stack:
            node = dag.nodes[stack.pop()]
            op = node["op"]
            if op == "and":
                stack.extend(int(a) for a in node["args"])
                continue
            if op not in ("le", "lt"):
                continue
            left, right = (dag.nodes[int(a)] for a in node["args"])
            if left["op"] == "var" and right["op"] == "const":
                bound = int(right["value"]) - (1 if op == "lt" else 0)
                var = int(left["var"])
                bounds[var] = min(bounds.get(var, bound), bound)
    return bounds


def input_variables(dag: TermDag, record: dict[str, Any], term: int) -> set[int]:
    """The variables the term depends on, through the range bounds of the
    variables it meets, to a fixpoint."""
    variables = record["variables"]
    seen: set[int] = set()
    frontier = list(dag.variables(term))
    while frontier:
        var = frontier.pop()
        if var in seen:
            continue
        seen.add(var)
        for side in ("lo", "hi"):
            bound = variables[var].get(side)
            if bound is not None:
                frontier.extend(dag.variables(int(bound)))
    return seen


def declared_bound(dag: TermDag, record: dict[str, Any], term: int,
                   preconditions: dict[int, int]) -> tuple[int | None, str]:
    """The declared bound of a term and the variable that gave it
    (module docstring). `(None, "")` when the term depends on no declared
    input, or on one with no declared maximum."""
    best: int | None = None
    name = ""
    found = False
    for var in sorted(input_variables(dag, record, term)):
        variable = record["variables"][var]
        if variable["origin"]["kind"] not in DECLARED_ORIGINS:
            continue
        found = True
        hi = closed_value(dag, variable.get("hi"))
        if var in preconditions:
            hi = preconditions[var] if hi is None else min(hi, preconditions[var])
        if hi is None:
            return None, variable["name"]
        if best is None or hi > best:
            best, name = hi, variable["name"]
    return (best, name) if found else (None, "")


def rows(entry: Entry) -> list[dict[str, Any]]:
    """One row per MAX answer a live implementation node names."""
    if not entry.accepted or entry.phase0 is None or entry.phase1 is None:
        return []
    record, result = entry.phase0, entry.phase1
    dag = entry.terms()
    preconditions = precondition_bounds(dag, record)
    answers = result["answers"]
    users: dict[int, list[dict[str, Any]]] = {}
    for node in result["implementations"]:
        if node["dropped"]:
            continue
        for at in node["answers"]:
            if answers[at]["kind"] == "MAX":
                users.setdefault(at, []).append(node)
    out = []
    for at in sorted(users):
        answer = answers[at]
        derived = int(answer["value"]) if answer["value"] is not None else None
        trivial = int(answer["interval_bound"]) if answer["interval_bound"] is not None else None
        declared, by = declared_bound(dag, record, int(answer["term"][0]), preconditions)
        rules = sorted({f"{n['rule']}{'.' + n['variant'] if n['variant'] else ''}"
                        for n in users[at]})
        out.append({
            "label": entry.label,
            "answer": at,
            "rules": rules,
            "nodes": len(users[at]),
            "term": dag.pretty(int(answer["term"][0]))[:60],
            "trivial": trivial,
            "derived": derived,
            "declared": declared,
            "declared_by": by,
            "layer": answer["layer"],
            "tight": bool(answer["tight"]),
            "below_trivial": derived is not None and trivial is not None and derived < trivial,
            "below_declared": derived is not None and declared is not None and derived < declared,
            "factor": (trivial / derived if derived and trivial is not None and derived > 0
                       else None),
        })
    return out


def summary(table: list[dict[str, Any]]) -> dict[str, Any]:
    with_bound = [r for r in table if r["derived"] is not None]
    below = [r for r in with_bound if r["below_trivial"]]
    equal = [r for r in with_bound if r["trivial"] is not None and r["derived"] == r["trivial"]]
    return {
        "capacities": len(table),
        "circuits": len({r["label"] for r in table}),
        "no_bound": len(table) - len(with_bound),
        "trivial_unbounded": sum(1 for r in with_bound if r["trivial"] is None),
        "equal_trivial": len(equal),
        "below_trivial": len(below),
        "below_trivial_labels": sorted({r["label"] for r in below}),
        "factors": sorted((r["factor"] for r in below if r["factor"] is not None), reverse=True),
        "below_declared": sum(1 for r in with_bound if r["below_declared"]),
        "equal_declared": sum(1 for r in with_bound
                              if r["declared"] is not None and r["derived"] == r["declared"]),
        "no_declared": sum(1 for r in with_bound if r["declared"] is None),
        "tight": sum(1 for r in with_bound if r["tight"]),
        "by_layer": {layer: sum(1 for r in table if r["layer"] == layer)
                     for layer in ("syntactic", "interval", "z3")},
    }


def table(source: Iterable[Entry]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for entry in source:
        out.extend(rows(entry))
    return out


def main() -> int:
    result = table(entries(CACHE_DIR))
    header = (f"{'circuit':<32}{'rules':<28}{'term':<24}{'trivial':>22}{'derived':>9}"
              f"{'declared':>10}{'by':<10}{'layer':>9}{'tight':>6}")
    print(header)
    print("-" * len(header))
    for row in result:
        show = lambda v: "-" if v is None else str(v)  # noqa: E731
        print(f"{row['label'][:31]:<32}{', '.join(row['rules'])[:27]:<28}{row['term'][:23]:<24}"
              f"{show(row['trivial']):>22}{show(row['derived']):>9}{show(row['declared']):>10}"
              f" {row['declared_by'][:9]:<9}{row['layer']:>9}{'yes' if row['tight'] else 'no':>6}")
    totals = summary(result)
    print("-" * len(header))
    print(f"{totals['capacities']} capacities over {totals['circuits']} circuits: "
          f"{totals['equal_trivial']} equal the trivial bound, {totals['below_trivial']} are "
          f"below it (factors {', '.join(f'{f:.1f}' for f in totals['factors'][:8])}), "
          f"{totals['trivial_unbounded']} have no trivial bound; "
          f"{totals['below_declared']} below the declared bound, "
          f"{totals['equal_declared']} equal to it, {totals['no_declared']} with none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
