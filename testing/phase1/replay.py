"""Evidence replay (`revision/phase1/acceptance.md` §2.7; `revision/README.md`
§5.6: "phase 3 replays them").

For a sample of the answers in A of every accepted circuit, the answer is
recomputed here, on the other side of the export, and compared with what
the oracle recorded:

- every answer of the *interval* layer is replayed by interval arithmetic
  over the `var_range` constraints of Φ, a Python mirror of
  `src/phase0/interval.rs` and of the three-valued evaluation of
  `src/phase1/oracle/layers.rs` (:class:`Intervals`); a MAX or MIN must
  give the recorded bound and the recorded `lo` / `hi`, a CONST the one
  value, a PROVE the entailment, a CANFAIL the refutation;
- up to `N` answers of the *Z3* layer per circuit, spread over the kinds,
  are replayed with `testing/phase0/phi_solver.py`'s arena over the
  constraints of Φ_e — the ranges, preconditions and declarations, the
  product facts, and the checks at positions ≤ e — under the answer's
  condition g. A PROVE "proved" must leave Φ_e ∧ g ∧ ¬P unsatisfiable, a
  CONST "constant v" Φ_e ∧ g ∧ e ≠ v, a CANFAIL "cannot fail"
  Φ_{p−1} ∧ φ ∧ ¬defcond, and a MAX "bound b" Φ_e ∧ g ∧ e > b (a MIN,
  e < b). The oracle asked over the dependency cone of the question, which
  is a subset of Φ_e, so an unsatisfiable answer must stay unsatisfiable
  here: a model is a *disagreement*, which is a defect. A satisfiable
  answer — "not proved", "can fail", "not constant" — that comes back
  unsatisfiable over the whole of Φ_e is not a disagreement about the
  answer but says that Φ_e has an unsatisfiable component outside the
  cone; it is counted as *weaker* and reported.

MAX and MIN are not among the three kinds `acceptance.md` names for the Z3
replay; they are replayed here as well because a capacity that is not a
bound is the one answer whose defect makes a circuit unsound.

```text
.venv/bin/python -m testing.phase1.replay [--per-circuit N]
```

`--per-circuit 0` replays every Z3 answer of every circuit; over the whole
parity set that is some twenty minutes.
"""

from __future__ import annotations

import argparse
from typing import Any, Iterable

from testing.phase0.terms import TermDag
from testing.phase1.cache import CACHE_DIR, Entry, entries

#: Z3 answers replayed per circuit unless `--per-circuit` says otherwise.
PER_CIRCUIT = 40

#: Milliseconds one replayed query may take before it is `unknown`.
QUERY_TIMEOUT_MS = 10_000

#: Arenas larger than this are not rebuilt in Z3 (`phi_solver.MAX_ARENA_NODES`
#: has the same role); the circuit's Z3 answers are then reported as skipped.
MAX_ARENA_NODES = 400_000

KINDS = ("MAX", "MIN", "PROVE", "CONST", "CANFAIL")


# ---------------------------------------------------------------------------
# The interval layer, mirrored
# ---------------------------------------------------------------------------


def _add(a: int | None, b: int | None) -> int | None:
    return None if a is None or b is None else a + b


def _min_lo(a: int | None, b: int | None) -> int | None:
    return None if a is None or b is None else min(a, b)


def _min_hi(a: int | None, b: int | None) -> int | None:
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


def _max_lo(a: int | None, b: int | None) -> int | None:
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


def _max_hi(a: int | None, b: int | None) -> int | None:
    return None if a is None or b is None else max(a, b)


class Intervals:
    """Interval arithmetic over a term DAG and the ranges of Φ
    (`src/phase0/interval.rs`), and the three-valued Boolean evaluation of
    `src/phase1/oracle/layers.rs` on top of it. `None` is the infinity on
    the side it appears."""

    def __init__(self, dag: TermDag, record: dict[str, Any]) -> None:
        self.dag = dag
        self.ranges: dict[int, tuple[int | None, int | None]] = {
            int(v["id"]): (v.get("lo"), v.get("hi")) for v in record["variables"]
        }
        self.memo: dict[int, tuple[int | None, int | None]] = {}

    def of(self, node_id: int) -> tuple[int | None, int | None]:
        node_id = int(node_id)
        if node_id in self.memo:
            return self.memo[node_id]
        # Reserved as unbounded first, so that a range that mentions its own
        # variable terminates (the Rust does the same).
        self.memo[node_id] = (None, None)
        value = self._compute(self.dag.nodes[node_id])
        self.memo[node_id] = value
        return value

    def _compute(self, node: dict[str, Any]) -> tuple[int | None, int | None]:
        op = node["op"]
        args = [int(a) for a in node.get("args", ())]
        if op == "const":
            n = int(node["value"])
            return (n, n)
        if op == "var":
            lo_node, hi_node = self.ranges.get(int(node["var"]), (None, None))
            lo = self.of(lo_node)[0] if lo_node is not None else None
            hi = self.of(hi_node)[1] if hi_node is not None else None
            return (lo, hi)
        if op == "add":
            a, b = self.of(args[0]), self.of(args[1])
            return (_add(a[0], b[0]), _add(a[1], b[1]))
        if op == "sub":
            a, b = self.of(args[0]), self.of(args[1])
            return (None if a[0] is None or b[1] is None else a[0] - b[1],
                    None if a[1] is None or b[0] is None else a[1] - b[0])
        if op == "mul":
            k = int(node["k"])
            lo, hi = self.of(args[0])
            if k < 0:
                lo, hi = hi, lo
            return (None if lo is None else lo * k, None if hi is None else hi * k)
        if op == "div":
            k = int(node["k"])
            if k == 0:
                return (None, None)
            lo, hi = self.of(args[0])
            if k < 0:
                lo, hi = hi, lo
            return (None if lo is None else lo // k, None if hi is None else hi // k)
        if op == "mod":
            k = int(node["k"])
            if k > 0:
                return (0, k - 1)
            if k < 0:
                return (k + 1, 0)
            return (None, None)
        if op == "min":
            a, b = self.of(args[0]), self.of(args[1])
            return (_min_lo(a[0], b[0]), _min_hi(a[1], b[1]))
        if op == "max":
            a, b = self.of(args[0]), self.of(args[1])
            return (_max_lo(a[0], b[0]), _max_hi(a[1], b[1]))
        if op == "ite":
            a, b = self.of(args[1]), self.of(args[2])
            return (_min_lo(a[0], b[0]), _max_hi(a[1], b[1]))
        return (None, None)

    # ── three-valued Booleans (`layers.rs::evaluate_bool`) ────────────

    def truth(self, node_id: int) -> bool | None:
        """`True`, `False`, or `None` for unknown."""
        node = self.dag.nodes[int(node_id)]
        op = node["op"]
        args = [int(a) for a in node.get("args", ())]
        if op == "true":
            return True
        if op == "false":
            return False
        if op == "bvar":
            return None
        if op == "not":
            inner = self.truth(args[0])
            return None if inner is None else not inner
        if op == "and":
            a, b = self.truth(args[0]), self.truth(args[1])
            if a is False or b is False:
                return False
            return True if a is True and b is True else None
        if op == "or":
            a, b = self.truth(args[0]), self.truth(args[1])
            if a is True or b is True:
                return True
            return False if a is False and b is False else None
        if op in ("lt", "le", "eq"):
            x, y = self.of(args[0]), self.of(args[1])
            if op == "lt":
                return _order(x, y, False)
            if op == "le":
                return _order(x, y, True)
            first, second = _order(x, y, False), _order(y, x, False)
            if first is True or second is True:
                return False
            if first is False and second is False:
                if x[0] is not None and x[0] == x[1] == y[0] == y[1]:
                    return True
            return None
        return None


def _order(x: tuple[int | None, int | None], y: tuple[int | None, int | None],
           inclusive: bool) -> bool | None:
    """`x < y` (or `x ≤ y`) decided by the two intervals (`layers.rs::order`)."""
    def strict(a: int, b: int) -> bool:
        return a <= b if inclusive else a < b
    if x[1] is not None and y[0] is not None and strict(x[1], y[0]):
        return True
    if x[0] is not None and y[1] is not None and not strict(x[0], y[1]):
        return False
    return None


def replay_interval(intervals: Intervals, answer: dict[str, Any]) -> tuple[str, str]:
    """(`agree` | `disagree` | `unreplayable`, detail) for one interval-layer
    answer."""
    kind, verdict = answer["kind"], answer["verdict"]
    term = int(answer["term"][0])
    if kind in ("MAX", "MIN"):
        lo, hi = intervals.of(term)
        mine = hi if kind == "MAX" else lo
        value = int(answer["value"]) if answer["value"] is not None else None
        evidence = answer["evidence"]
        recorded = (None if evidence.get("lo") is None else int(evidence["lo"]),
                    None if evidence.get("hi") is None else int(evidence["hi"]))
        if mine == value and (evidence.get("kind") != "interval" or recorded == (lo, hi)):
            return "agree", f"{kind} {value}"
        return "disagree", f"{kind}: recorded {value} with {recorded}, replayed {(lo, hi)}"
    if kind == "CONST":
        if verdict == "constant":
            lo, hi = intervals.of(term)
            value = int(answer["value"])
            if lo == hi == value:
                return "agree", f"constant {value}"
            return "disagree", f"CONST: recorded constant {value}, replayed {(lo, hi)}"
        if verdict == "constant_bool":
            truth = intervals.truth(term)
            value = bool(int(answer["value"]))
            if truth is value:
                return "agree", f"constant_bool {value}"
            return "disagree", f"CONST: recorded constant_bool {value}, replayed {truth}"
        if verdict == "not_constant":
            node = intervals.dag.nodes[term]
            if node.get("sort") == "bool":
                truth = intervals.truth(term)
                if truth is None:
                    return "agree", "not constant"
                return "disagree", f"CONST: recorded not_constant, replayed {truth}"
            lo, hi = intervals.of(term)
            if lo is None or hi is None or lo != hi:
                return "agree", "not constant"
            return "disagree", f"CONST: recorded not_constant, replayed {(lo, hi)}"
    if kind == "PROVE":
        truth = intervals.truth(term)
        if verdict == "proved":
            return ("agree", "proved") if truth is True else (
                "disagree", f"PROVE: recorded proved, replayed {truth}")
        return ("agree", "not proved") if truth is not True else (
            "disagree", "PROVE: recorded not_proved, replayed True")
    if kind == "CANFAIL":
        path, defcond = (int(t) for t in answer["term"])
        taken, holds = intervals.truth(path), intervals.truth(defcond)
        failing = None if holds is None else not holds
        refuted = taken is False or failing is False
        if verdict == "cannot_fail":
            return ("agree", "cannot fail") if refuted else (
                "disagree", f"CANFAIL: recorded cannot_fail, replayed path {taken}, "
                            f"defcond {holds}")
        return ("agree", "can fail") if not refuted else (
            "disagree", "CANFAIL: recorded can_fail, replayed refuted")
    return "unreplayable", f"{kind} {verdict}"


# ---------------------------------------------------------------------------
# The Z3 layer, replayed over Φ_e
# ---------------------------------------------------------------------------


class PhiE:
    """Φ_e in Z3 over the shared arena: the base of Φ asserted once, the
    checks and the question pushed per query."""

    def __init__(self, dag: TermDag, record: dict[str, Any]) -> None:
        import z3

        from testing.phase0.phi_solver import Z3Arena

        self.z3 = z3
        self.arena = Z3Arena(dag)
        self.checks: list[Any] = []
        base: list[Any] = []
        for constraint in record["constraints"]:
            formula = self.arena[int(constraint["formula"])]
            (self.checks if constraint["kind"] == "check" else base).append(formula)
        for fact in record.get("products", []):
            right: Any = z3.IntVal(1)
            for factor in fact["factors"]:
                right = right * self.arena[int(factor)]
            base.append(self.arena.variables[int(fact["var"])] == right)
        self.solver = z3.Solver()
        self.solver.set("timeout", QUERY_TIMEOUT_MS)
        self.solver.add(*base)

    def check(self, epoch: int, assumptions: list[Any]) -> str:
        """`sat`, `unsat` or `unknown` for Φ_epoch ∧ assumptions."""
        self.solver.push()
        try:
            self.solver.add(*self.checks[:epoch])
            result = self.solver.check(*assumptions)
        finally:
            self.solver.pop()
        if result == self.z3.sat:
            return "sat"
        if result == self.z3.unsat:
            return "unsat"
        return "unknown"


def replay_z3(phi: PhiE, answer: dict[str, Any]) -> tuple[str, str]:
    """(`agree` | `disagree` | `weaker` | `unknown` | `unreplayable`, detail)
    for one Z3-layer answer, per the module docstring."""
    z3 = phi.z3
    kind, verdict = answer["kind"], answer["verdict"]
    epoch = int(answer["epoch"])
    g = phi.arena[int(answer["condition"])]
    if kind == "CANFAIL":
        path, defcond = (phi.arena[int(t)] for t in answer["term"])
        outcome = phi.check(epoch, [path, z3.Not(defcond)])
        expected = "unsat" if verdict == "cannot_fail" else "sat"
    elif kind == "PROVE":
        claim = phi.arena[int(answer["term"][0])]
        outcome = phi.check(epoch, [g, z3.Not(claim)])
        expected = "unsat" if verdict == "proved" else "sat"
    elif kind == "CONST":
        term = phi.arena[int(answer["term"][0])]
        if verdict == "constant":
            outcome = phi.check(epoch, [g, term != z3.IntVal(int(answer["value"]))])
            expected = "unsat"
        elif verdict == "constant_bool":
            outcome = phi.check(epoch, [g, term != z3.BoolVal(bool(int(answer["value"])))])
            expected = "unsat"
        else:
            # Not constant: two models with different values exist. One
            # model is enough to show the question was not vacuous.
            outcome = phi.check(epoch, [g])
            expected = "sat"
    elif kind in ("MAX", "MIN"):
        if answer["value"] is None:
            return "unreplayable", f"{kind} without a bound"
        term = phi.arena[int(answer["term"][0])]
        bound = z3.IntVal(int(answer["value"]))
        outcome = phi.check(epoch, [g, term > bound if kind == "MAX" else term < bound])
        expected = "unsat"
    else:
        return "unreplayable", f"{kind} {verdict}"
    if outcome == expected:
        return "agree", f"{kind} {verdict}"
    if outcome == "unknown":
        return "unknown", f"{kind} {verdict}: Z3 timed out on the replay"
    if expected == "sat" and outcome == "unsat":
        return "weaker", f"{kind} {verdict}: unsatisfiable over the whole of Φ_e"
    return "disagree", f"{kind} {verdict}: replay over Φ_e is {outcome}"


#: The `why` texts of a CONST "not constant" answer: the oracle records the
#: two models it found, or a model that gave the term no value, under the
#: `unknown` evidence variant (`oracle/mod.rs`), so the variant alone does
#: not tell a decided answer from a timeout.
NOT_CONSTANT_WHYS = ("both values have a model", "the model gave no integer for the term")


def timed_out(answer: dict[str, Any]) -> bool:
    """Whether the oracle recorded no answer for lack of time or budget."""
    evidence = answer["evidence"]
    return evidence["kind"] == "unknown" and evidence.get("why") not in NOT_CONSTANT_WHYS


def sample(answers: list[dict[str, Any]], per_circuit: int) -> list[dict[str, Any]]:
    """Up to `per_circuit` Z3 answers, one kind at a time in A order, so
    that every kind present is represented."""
    by_kind: dict[str, list[dict[str, Any]]] = {kind: [] for kind in KINDS}
    for answer in answers:
        if answer["layer"] == "z3" and not timed_out(answer):
            by_kind[answer["kind"]].append(answer)
    if per_circuit <= 0:
        per_circuit = sum(len(found) for found in by_kind.values())
    chosen: list[dict[str, Any]] = []
    cursors = {kind: 0 for kind in KINDS}
    while len(chosen) < per_circuit:
        progressed = False
        for kind in KINDS:
            if cursors[kind] < len(by_kind[kind]) and len(chosen) < per_circuit:
                chosen.append(by_kind[kind][cursors[kind]])
                cursors[kind] += 1
                progressed = True
        if not progressed:
            break
    return chosen


def rows(entry: Entry, per_circuit: int = PER_CIRCUIT) -> list[dict[str, Any]]:
    """One row per replayed answer of an accepted circuit."""
    if not entry.accepted or entry.phase0 is None or entry.phase1 is None:
        return []
    dag = entry.terms()
    record, result = entry.phase0, entry.phase1
    out: list[dict[str, Any]] = []
    intervals = Intervals(dag, record)
    for answer in result["answers"]:
        if answer["layer"] != "interval":
            continue
        outcome, detail = replay_interval(intervals, answer)
        out.append({"label": entry.label, "answer": answer["index"], "layer": "interval",
                    "kind": answer["kind"], "verdict": answer["verdict"],
                    "outcome": outcome, "detail": detail})
    chosen = sample(result["answers"], per_circuit)
    if not chosen:
        return out
    if len(dag.nodes) > MAX_ARENA_NODES:
        for answer in chosen:
            out.append({"label": entry.label, "answer": answer["index"], "layer": "z3",
                        "kind": answer["kind"], "verdict": answer["verdict"],
                        "outcome": "skipped",
                        "detail": f"arena of {len(dag.nodes)} nodes not rebuilt"})
        return out
    phi = PhiE(dag, record)
    for answer in chosen:
        outcome, detail = replay_z3(phi, answer)
        out.append({"label": entry.label, "answer": answer["index"], "layer": "z3",
                    "kind": answer["kind"], "verdict": answer["verdict"],
                    "outcome": outcome, "detail": detail})
    return out


def summary(table: list[dict[str, Any]]) -> dict[str, Any]:
    def count(layer: str, outcome: str) -> int:
        return sum(1 for r in table if r["layer"] == layer and r["outcome"] == outcome)
    return {
        "replayed": len(table),
        "interval": sum(1 for r in table if r["layer"] == "interval"),
        "z3": sum(1 for r in table if r["layer"] == "z3"),
        "agree": sum(1 for r in table if r["outcome"] == "agree"),
        "disagree": sum(1 for r in table if r["outcome"] == "disagree"),
        "weaker": count("z3", "weaker"),
        "unknown": count("z3", "unknown"),
        "skipped": count("z3", "skipped"),
        "unreplayable": sum(1 for r in table if r["outcome"] == "unreplayable"),
        "by_kind": {kind: {"interval": sum(1 for r in table if r["layer"] == "interval"
                                            and r["kind"] == kind),
                           "z3": sum(1 for r in table if r["layer"] == "z3"
                                      and r["kind"] == kind)}
                    for kind in KINDS},
        "disagreements": [r for r in table if r["outcome"] == "disagree"],
        "weaker_rows": [r for r in table if r["outcome"] == "weaker"],
        "circuits": len({r["label"] for r in table}),
    }


def table(source: Iterable[Entry], per_circuit: int = PER_CIRCUIT) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for entry in source:
        out.extend(rows(entry, per_circuit))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-circuit", type=int, default=PER_CIRCUIT,
                        help="Z3 answers replayed per circuit")
    arguments = parser.parse_args()
    result = table(entries(CACHE_DIR), arguments.per_circuit)
    totals = summary(result)
    for row in totals["disagreements"]:
        print(f"  DISAGREE {row['label']} answer {row['answer']} ({row['layer']}): {row['detail']}")
    for row in totals["weaker_rows"][:20]:
        print(f"  weaker   {row['label']} answer {row['answer']}: {row['detail']}")
    print(f"{totals['replayed']} answers replayed over {totals['circuits']} circuits: "
          f"{totals['interval']} interval, {totals['z3']} Z3; {totals['agree']} agree, "
          f"{totals['disagree']} disagree, {totals['weaker']} weaker, "
          f"{totals['unknown']} unknown, {totals['skipped']} skipped, "
          f"{totals['unreplayable']} unreplayable")
    return 1 if totals["disagree"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
