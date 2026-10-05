"""Acceptance and invariants (`revision/phase1/acceptance.md` §2.1).

Two things, both from the cache:

1. For every accepted circuit, the invariants of `revision/phase1/README.md`
   §2 re-checked on the exported JSON, independently of the Rust checker in
   `src/phase1/graph.rs`:

   - G is acyclic;
   - every requirement reachable from a root has at least one implementation
     node;
   - every implementation node's inputs and outputs are requirement nodes of
     G, and every provider names the requirement it answers;
   - every check of Φ is an `Enforce` root or has a CANFAIL answer "cannot
     fail" in A;
   - every answer a node names is in A (the other direction — A holds no
     question no node asked — needs the list of the sweep's and the dropped
     premises' answers, which the export does not carry; the count of
     answers no node names is reported instead);
   - no parameter mentions a variable: every parameter is a decimal integer
     or an integer range;
   - every node below an `Enforce(k)` root has epoch < position(k).

2. The acceptance count against phase 0's 150, and every circuit phase 1
   rejects with its reason in one of four categories (`acceptance.md` §2.1):
   *phase 0 record gap*, *outside the library by decision*, *design gap*,
   *rule gap*. The category is assigned by :func:`category_of` from the
   rejection message, and the table there says which pattern means which.

```text
.venv/bin/python -m testing.phase1.invariants
```
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable

from testing.phase1.cache import CACHE_DIR, Entry, entries

#: The four categories of `acceptance.md` §2.1, in report order.
CATEGORIES = (
    "phase 0 record gap",
    "outside the library by decision",
    "design gap",
    "rule gap",
)

#: Rejections the message alone does not place, decided by reading the
#: kernel against `rules/README.md` §9's not-covered list: the message names
#: the phase 0 rule that produced the value, and §9 excludes the operation
#: by a finer criterion (how many positions, what the operand is) or under
#: another name (the comprehension behind an `E-Stack`).
DECIDED: dict[str, tuple[str, str]] = {
    "ds1000_case0300::kernel": (
        "outside the library by decision",
        "`np.percentile` is in §9's not-covered list"),
    "ds1000_case0344::kernel": (
        "outside the library by decision",
        "`np.array(list(s))`, an `np.array` of a composite through its `E-Copy` "
        "wrapper, is in §9's not-covered list"),
    "ds1000_case0361::kernel": (
        "outside the library by decision",
        "`np.delete` at several positions is in §9's not-covered list"),
    "ds1000_case0469::kernel": (
        "outside the library by decision",
        "Complex arithmetic (`1j * np.imag(s)`) is in §9's not-covered list"),
    "ds1000_case0473::kernel": (
        "outside the library by decision",
        "`any(np.array_equal(c, x) for x in CNTS)`: `E-Comp` and `array_equal` "
        "are in §9's not-covered list"),
    "ds1000_case0373::kernel": (
        "phase 0 record gap",
        "a chip call's result is recorded as `Applied(S-Call)` over the callee's "
        "values rather than as an alias of the returned value, so §9's \"no rule: "
        "aliases are phase 0's\" cannot apply to it"),
}

_INTEGER = re.compile(r"^-?\d+$")


# ---------------------------------------------------------------------------
# The invariants of `revision/phase1/README.md` §2, on the exported JSON
# ---------------------------------------------------------------------------


def violations(entry: Entry) -> list[str]:
    """Every invariant of §2 that the exported record of `entry` breaks."""
    if entry.phase1 is None or entry.phase0 is None:
        return []
    graph = entry.phase1
    found: list[str] = []
    found.extend(check_edges(graph))
    found.extend(check_acyclic(graph))
    found.extend(check_providers(graph))
    found.extend(check_checks_enforced(entry))
    found.extend(check_answers_named(graph))
    found.extend(check_parameters(graph))
    found.extend(check_epoch_filter(graph))
    return found


def check_edges(graph: dict[str, Any]) -> list[str]:
    """Every edge lands on a node of G; every provider outputs what it
    answers; every root is a requirement."""
    found = []
    requirements, implementations = graph["requirements"], graph["implementations"]
    for node in implementations:
        for edge in (*node["inputs"], *node["outputs"]):
            if not 0 <= edge < len(requirements):
                found.append(f"edges: implementation {node['index']} names requirement {edge}")
    for requirement in requirements:
        for provider in requirement["providers"]:
            if not 0 <= provider < len(implementations):
                found.append(f"edges: requirement {requirement['index']} names "
                             f"implementation {provider}")
                continue
            if requirement["index"] not in implementations[provider]["outputs"]:
                found.append(f"edges: requirement {requirement['index']} names "
                             f"implementation {provider}, which does not output it")
            if implementations[provider]["dropped"]:
                found.append(f"edges: requirement {requirement['index']} names dropped "
                             f"implementation {provider}")
    for root in graph["roots"]:
        if not 0 <= root < len(requirements):
            found.append(f"roots: {root} is not a requirement of G")
    return found


def successors(graph: dict[str, Any], at: int) -> list[int]:
    """The requirements below `at`: the inputs of its providers."""
    implementations = graph["implementations"]
    out: set[int] = set()
    for provider in graph["requirements"][at]["providers"]:
        out.update(implementations[provider]["inputs"])
    return sorted(out)


def check_acyclic(graph: dict[str, Any]) -> list[str]:
    """G is acyclic: a depth-first search over requirement → provider →
    input finds no back edge."""
    OPEN, DONE = 1, 2
    mark: dict[int, int] = {}
    found = []
    for start in range(len(graph["requirements"])):
        if start in mark:
            continue
        stack = [(start, iter(successors(graph, start)))]
        mark[start] = OPEN
        while stack:
            at, children = stack[-1]
            child = next(children, None)
            if child is None:
                mark[at] = DONE
                stack.pop()
                continue
            state = mark.get(child)
            if state == OPEN:
                found.append(f"acyclic: requirement {at} reaches itself through {child}")
            elif state is None:
                mark[child] = OPEN
                stack.append((child, iter(successors(graph, child))))
    return found


def reachable(graph: dict[str, Any]) -> set[int]:
    """The requirements reachable from the roots."""
    seen: set[int] = set()
    stack = list(graph["roots"])
    while stack:
        at = stack.pop()
        if at in seen:
            continue
        seen.add(at)
        stack.extend(successors(graph, at))
    return seen


def check_providers(graph: dict[str, Any]) -> list[str]:
    """Every requirement reachable from a root has an implementation node."""
    found = []
    for at in sorted(reachable(graph)):
        requirement = graph["requirements"][at]
        if not requirement["providers"]:
            found.append(f"providers: requirement {at} ({requirement['form']} "
                         f"{requirement['target']}) has no implementation node")
    return found


def unreached_without_provider(graph: dict[str, Any]) -> int:
    """Requirements no root reaches and no node provides: rows a consumer
    asked and did not need (`Phase1Result::missing`), reported as a count."""
    live = reachable(graph)
    return sum(1 for r in graph["requirements"]
               if r["index"] not in live and not r["providers"])


def enforce_roots(graph: dict[str, Any]) -> set[int]:
    """The constraint indices of Φ with an `Enforce` root."""
    roots = set(graph["roots"])
    out = set()
    for requirement in graph["requirements"]:
        if requirement["form"] != "enforce":
            continue
        constraint = int(requirement["target"].split()[-1])
        if requirement["index"] in roots:
            out.add(constraint)
    return out


def check_checks_enforced(entry: Entry) -> list[str]:
    """Every check of Φ is an `Enforce` root or has a CANFAIL answer "cannot
    fail" in A. The CANFAIL of check k at position p is keyed by its (path,
    defcond) terms at epoch p − 1 (`src/phase1/oracle/sweep.rs`)."""
    graph = entry.phase1
    assert graph is not None
    enforced = enforce_roots(graph)
    cannot_fail: set[tuple[int, int, int]] = set()
    for answer in graph["answers"]:
        if answer["kind"] == "CANFAIL" and answer["verdict"] == "cannot_fail":
            path, defcond = answer["term"]
            cannot_fail.add((int(path), int(defcond), int(answer["epoch"])))
    found = []
    for position, (at, check) in enumerate(entry.checks(), start=1):
        if at in enforced:
            continue
        key = (int(check["path"]), int(check["defcond"]), position - 1)
        if key not in cannot_fail:
            found.append(f"checks: check {position} (constraint {at}) is neither "
                         f"enforced nor answered cannot_fail")
    for requirement in graph["requirements"]:
        if requirement["form"] == "enforce" and requirement["index"] not in graph["roots"]:
            found.append(f"checks: Enforce requirement {requirement['index']} is not a root")
    return found


def check_answers_named(graph: dict[str, Any]) -> list[str]:
    """Every answer a node names is in A."""
    found = []
    count = len(graph["answers"])
    for node in graph["implementations"]:
        for at in node["answers"]:
            if not 0 <= at < count:
                found.append(f"answers: implementation {node['index']} names answer {at}, "
                             f"A holds {count}")
    return found


def unnamed_answers(graph: dict[str, Any]) -> Counter:
    """Answers in A no implementation node names, by kind: the CANFAIL
    sweep's and the premises of rule instances that produced no node."""
    named: set[int] = set()
    for node in graph["implementations"]:
        named.update(node["answers"])
    return Counter(a["kind"] for a in graph["answers"] if a["index"] not in named)


def check_parameters(graph: dict[str, Any]) -> list[str]:
    """No parameter mentions a variable: every one is an integer or an
    integer range (`graph::Parameter`)."""
    found = []
    for node in graph["implementations"]:
        for name, value in node["parameters"].items():
            ok = (isinstance(value, str) and _INTEGER.match(value)) or (
                isinstance(value, dict) and set(value) == {"lo", "hi"}
                and all(isinstance(value[k], str) and _INTEGER.match(value[k])
                        for k in ("lo", "hi"))
            )
            if not ok:
                found.append(f"parameters: {node['rule']} {node['index']} has "
                             f"{name} = {value!r}")
    return found


def check_epoch_filter(graph: dict[str, Any]) -> list[str]:
    """Every node below an `Enforce(k)` root has epoch < position(k)."""
    found = []
    requirements, implementations = graph["requirements"], graph["implementations"]
    for root in graph["roots"]:
        position = requirements[root]["filter"]["epoch_below"]
        if position is None:
            continue
        seen: set[int] = set()
        stack = [root]
        while stack:
            at = stack.pop()
            if at in seen:
                continue
            seen.add(at)
            for provider in requirements[at]["providers"]:
                node = implementations[provider]
                if node["epoch"] >= position:
                    found.append(f"epochs: {node['rule']} {provider} at epoch "
                                 f"{node['epoch']} provides a wire below Enforce at "
                                 f"position {position}")
                stack.extend(node["inputs"])
    return found


# ---------------------------------------------------------------------------
# Rejections, by category
# ---------------------------------------------------------------------------


def category_of(entry: Entry) -> tuple[str, str]:
    """The category of a phase 1 rejection and one line saying what is
    missing. The patterns, in the order tried:

    | message | category | why |
    |---|---|---|
    | `a capacity of N slots is outside the values Ω admits` | phase 0 record gap | Φ bounds the term by the dtype range only: a `size_product` σ carries no relation to its factors (`rules/README.md` §10 item 12) |
    | `child_condition` | design gap | the consumer's g does not imply the child's ψ, syntactically or by PROVE; `README.md` §4 rejects |
    | `node_budget` | design gap | Ω's node budget |
    | a label in :data:`DECIDED` | as decided there | the operation is in §9's not-covered list under a finer criterion than the message shows, or the record shape is the cause |
    | `… produced by X and the library covers with no rule` | design gap | §9 assigns X to phase 0's keying and aliasing, and a value produced by it still reaches phase 1 as a value of its own |
    | `R has no row for the primitive/operator …` | rule gap | the rule exists and its row table lacks the operation |
    | `no rule provides F for K, which phase 0 produced by X and the library covers with R` | rule gap | §9 maps X to R, and R has no instance over this interface |
    """
    if entry.label in DECIDED:
        return DECIDED[entry.label]
    detail = entry.detail or ""
    reason = entry.reason or ""
    if "outside the values Ω admits" in detail:
        return ("phase 0 record gap",
                "a capacity bounded by the dtype range alone: σ carries no relation "
                "to its factors (`rules/README.md` §10 item 12)")
    if reason == "child_condition":
        return ("design gap", "a child is read under a condition that does not imply "
                              "its ψ (`README.md` §4, conditions of child requirements)")
    if reason == "node_budget":
        return ("design gap", "Ω's node budget")
    produced = re.search(r"produced by ([^;]*?) and the library covers with ([^;]*)", detail)
    if produced:
        source, covered = produced.group(1).strip(), produced.group(2).strip()
        if covered == "no rule":
            return ("design gap",
                    f"a value produced by {source}, which §9 leaves to phase 0's keys "
                    f"and aliases, reaches phase 1 as a value of its own")
        family = re.search(r"no rule provides (\w+) for", detail)
        return ("rule gap",
                f"{covered} has no instance providing {family.group(1) if family else '?'} "
                f"over a value of {source}")
    row = re.search(r"(\w+) has no row for the (primitive|operator) `([^`]*)` on (\w+)", detail)
    if row:
        return ("rule gap", f"{row.group(1)} has no row for `{row.group(3)}` on {row.group(4)}")
    return ("rule gap", detail[:120])


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


def rows(entry: Entry) -> dict[str, Any]:
    """One row per circuit of the parity set."""
    row: dict[str, Any] = {
        "label": entry.label,
        "family": entry.family,
        "status": entry.status,
        "reason": entry.reason or "",
        "detail": (entry.detail or "")[:200],
        "category": "",
        "line": "",
        "violations": [],
        "rust_violations": [],
        "unnamed_answers": {},
        "unreached_without_provider": 0,
    }
    if entry.accepted:
        assert entry.phase1 is not None
        row["violations"] = violations(entry)
        row["rust_violations"] = list(entry.phase1.get("violations", []))
        row["unnamed_answers"] = dict(unnamed_answers(entry.phase1))
        row["unreached_without_provider"] = unreached_without_provider(entry.phase1)
    elif entry.rejected_by_phase1:
        row["category"], row["line"] = category_of(entry)
    return row


def summary(table: list[dict[str, Any]]) -> dict[str, Any]:
    """The counts the report and the acceptance test read."""
    seen_by_phase1 = [r for r in table if r["status"] not in ("phase0_rejected",)]
    accepted = [r for r in table if r["status"] == "ok"]
    rejected = [r for r in table if r["status"] == "phase1_rejected"]
    failed = [r for r in table if r["status"] not in ("ok", "phase1_rejected", "phase0_rejected")]
    return {
        "parity": len(table),
        "phase0_rejected": sum(1 for r in table if r["status"] == "phase0_rejected"),
        "seen_by_phase1": len(seen_by_phase1),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "failed": len(failed),
        "failed_labels": [r["label"] for r in failed],
        "categories": Counter(r["category"] for r in rejected),
        "reasons": Counter(r["reason"] for r in rejected),
        "with_violations": [r["label"] for r in accepted if r["violations"]],
        "with_rust_violations": [r["label"] for r in accepted if r["rust_violations"]],
        "violation_kinds": Counter(v.split(":")[0] for r in accepted for v in r["violations"]),
        "unnamed_answers": sum((Counter(r["unnamed_answers"]) for r in accepted), Counter()),
        "unreached_without_provider": sum(r["unreached_without_provider"] for r in accepted),
    }


def table(source: Iterable[Entry]) -> list[dict[str, Any]]:
    return [rows(entry) for entry in source]


def main() -> int:
    result = table(entries(CACHE_DIR))
    for row in result:
        if row["status"] == "ok":
            mark = "pass" if not row["violations"] else "FAIL"
            print(f"  ok        {row['label']:<54} invariants {mark}")
            for violation in row["violations"][:5]:
                print(f"              {violation}")
        elif row["status"] == "phase1_rejected":
            print(f"  rejected  {row['label']:<54} [{row['category']}] {row['line'][:90]}")
        elif row["status"] != "phase0_rejected":
            print(f"  {row['status']:<9} {row['label']:<54} {row['detail'][:90]}")
    totals = summary(result)
    print()
    print(f"parity {totals['parity']}, phase 0 rejected {totals['phase0_rejected']}, "
          f"seen by phase 1 {totals['seen_by_phase1']}: accepted {totals['accepted']}, "
          f"rejected {totals['rejected']}, harness failures {totals['failed']}")
    for category in CATEGORIES:
        print(f"  {totals['categories'].get(category, 0):>3}  {category}")
    print(f"invariant violations on {len(totals['with_violations'])} circuits"
          + (": " + ", ".join(totals["with_violations"]) if totals["with_violations"] else ""))
    print(f"answers no node names, by kind: {dict(totals['unnamed_answers'])}; "
          f"unreached requirements without a provider: {totals['unreached_without_provider']}")
    return 1 if totals["with_violations"] or totals["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
