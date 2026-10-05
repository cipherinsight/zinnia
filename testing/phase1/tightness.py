"""Tightness verification (`revision/phase1/acceptance.md` §2.5;
`revision/README.md` §5.6 step 3 and §11, row 4: how often Φ misses a
relation).

For every MAX the oracle answered with Z3 and a model, in every accepted
circuit: build concrete inputs from the model, run the source with the
reference evaluator of `testing/phase0/refeval.py`, read θ(u) off the run
with `testing/phase0/adequacy.theta_of`, evaluate the MAX's term under it,
and compare with the bound.

**What the generator can realise.** A model is an assignment to Φ's
variables. The shape symbols and input lengths it names are set exactly
(`ModelDrawer`, over `testing/phase0/inputs.Drawer`), and so is a scalar
input the model names through a `data_scalar` variable. A *count* — the
number of true positions of a mask the program computes — a `unique_count`,
a `value_sized` length and a data Boolean depend on element values through
the program, and the generator has no inverse for that: it draws the
elements at random. So for each MAX several draws are run, the model-shaped
one first and then `DRAWS` random draws of the same shapes, and the largest
value the term reached on a *successful* run is compared with the bound.
A failing run does not count: a MAX bounds the successful executions
(`../README.md` §8, completeness).

Per MAX the row says `tight` (some run reached the bound), `loose` (no run
reached it, and the model names no variable the generator cannot target:
the bound is above what the program does, so Φ lacks a relation), or
`unrealised` (no run reached it and the model pins a count or a data
variable the generator cannot set — the verdict is not a measurement of Φ),
and `unsound` when a run exceeded the bound, which is a defect. A bound the
run exceeded, or a term the run could not evaluate, names the circuit.

```text
.venv/bin/python -m testing.phase1.tightness [--draws K]
```
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from testing.phase0 import REPO_ROOT
from testing.phase0.corpus import ChildLimits, Circuit, first_line, limit_address_space, parity_set, run_in_child
from testing.phase1.cache import CACHE_DIR, Entry, file_of, is_current

#: Random draws of the model's shapes per MAX, after the model-shaped one:
#: alternately from the benign element domain of `inputs.py` and from
#: :data:`WIDE`, so that a length that depends on an element value through
#: a `min` or a slice endpoint has a chance to reach its bound.
DRAWS = 16

#: The wide element domain, inclusive: enough to exceed every capacity the
#: corpus declares below it, small enough that a run mostly succeeds.
WIDE = 64

#: Address space and wall clock one circuit's child may use: the reference
#: evaluator unrolls what phase 0 unrolled.
ADDRESS_SPACE_LIMIT = 8 * 1024**3
TIMEOUT_SECONDS = 600.0

def parse_model(text: str) -> dict[str, int | bool]:
    """Z3's model as the oracle prints it: one `name -> value` per line."""
    out: dict[str, int | bool] = {}
    for line in text.splitlines():
        if "->" not in line:
            continue
        name, value = (part.strip() for part in line.split("->", 1))
        if value in ("true", "false"):
            out[name] = value == "true"
        else:
            try:
                out[name] = int(value.replace("(", "").replace(")", "").replace(" ", ""))
            except ValueError:
                continue
    return out


def model_assignment(record: dict[str, Any], model: dict[str, int | bool]) -> dict[int, int | bool]:
    """The model as variable id → value, through the printer's names."""
    by_name = {v["name"]: int(v["id"]) for v in record["terms"]["vars"]}
    return {by_name[name]: value for name, value in model.items() if name in by_name}


# ---------------------------------------------------------------------------
# Inputs shaped by a model
# ---------------------------------------------------------------------------


class ModelDrawer:
    """`inputs.Drawer` with the shapes and scalars a model names fixed.

    Built inside the child, where `testing.phase0.inputs` is imported.
    """

    def __init__(self, base: Any, symbols: dict[str, int], lengths: dict[tuple[str, int], int],
                 scalars: dict[str, int]) -> None:
        self.base = base
        self.fixed_symbols = symbols
        self.lengths = lengths
        self.scalars = scalars

    def tuple_for(self, specs: list[Any]) -> list[Any]:
        self.base.symbols = dict(self.fixed_symbols)
        return [self.value(spec.dt, spec) for spec in specs]

    def value(self, dt: Any, spec: Any) -> Any:
        if isinstance(dt, str):
            if dt == "Integer" and spec.key in self.scalars:
                return int(self.scalars[spec.key])
            return self.base.scalar(dt, spec)
        kind, body = next(iter(dt.items()))
        if kind == "DynamicNDArray":
            bounds = body.get("max_shape") or [body["max_length"]]
            shape = tuple(
                int(self.lengths.get((spec.name, axis),
                                     self.base.rng.integers(0, min(int(bound), self.base.settings.max_axis),
                                                            endpoint=True)))
                for axis, bound in enumerate(bounds)
            )
            return self.base.array(shape, body["dtype"])
        if kind in ("List", "Tuple"):
            elements = [self.value(element, spec) for element in body["elements"]]
            return elements if kind == "List" else tuple(elements)
        if kind == "PoseidonHashed":
            return self.value(body["dtype"], spec)
        return self.base.value(dt, spec)


def wide_drawer(base_class: Any) -> Any:
    """`inputs.Drawer` with integers drawn from `[-WIDE, WIDE]`, clamped into
    the range phase 0 recorded, built here because `inputs` is imported in
    the child."""

    class WideDrawer(base_class):
        def integer(self, bounds: tuple[int | None, int | None]) -> int:
            lo, hi = bounds
            low = -WIDE if lo is None else max(-WIDE, lo)
            high = WIDE if hi is None else min(WIDE, hi)
            if low > high:
                return low
            return int(self.rng.integers(low, high, endpoint=True))

    return WideDrawer


def targets_of(record: dict[str, Any], assignment: dict[int, int | bool]) -> dict[str, Any]:
    """What the generator sets from a model, and the variables it cannot."""
    symbols: dict[str, int] = {}
    lengths: dict[tuple[str, int], int] = {}
    scalars: dict[str, int] = {}
    unsettable: list[str] = []
    for var, value in assignment.items():
        variable = record["variables"][var]
        origin = variable["origin"]
        kind = origin["kind"]
        if kind == "shape_symbol":
            symbols[origin["symbol"]] = int(value)
        elif kind == "input_length":
            lengths[(origin["input"], int(origin["axis"]))] = int(value)
        elif kind == "data_scalar" and origin["value"]["key"].startswith("|||"):
            scalars[origin["value"]["key"]] = int(value)
        else:
            unsettable.append(variable["name"])
    return {"symbols": symbols, "lengths": lengths, "scalars": scalars,
            "unsettable": unsettable}


# ---------------------------------------------------------------------------
# The child: one circuit, every MAX with a model
# ---------------------------------------------------------------------------


def _check_circuit(kernel_path: str, function: str, cache_file: str, draws: int) -> list[dict[str, Any]]:
    from testing.phase0 import inputs as draw_inputs
    from testing.phase0.adequacy import Phase0Record, _disambiguate, theta_of
    from testing.phase0.corpus import load_module
    from testing.phase0.payloads import core_payloads, external_impls
    from testing.phase0.refeval import evaluate
    from testing.phase0.terms import UnboundVariable
    from zinnia import ZKCircuit

    entry = Entry.load(Path(cache_file))
    assert entry.phase0 is not None and entry.phase1 is not None
    record, result = entry.phase0, entry.phase1
    dag = entry.terms()
    phase0_record = Phase0Record.of(record)
    module = load_module(Path(kernel_path))
    zk_circuit = ZKCircuit.from_method(getattr(module, function))
    payloads = core_payloads(zk_circuit)
    impls = external_impls(zk_circuit)
    specs = draw_inputs.InputSpec.of(payloads[0])
    domain = draw_inputs.Domain.of(record)
    rows: list[dict[str, Any]] = []
    for answer in result["answers"]:
        if answer["kind"] != "MAX" or answer["layer"] != "z3" or answer["evidence"]["kind"] != "model":
            continue
        if answer["value"] is None:
            continue
        bound = int(answer["value"])
        term = int(answer["term"][0])
        assignment = model_assignment(record, parse_model(answer["evidence"]["model"]))
        targets = targets_of(record, assignment)
        row: dict[str, Any] = {
            "label": entry.label,
            "answer": answer["index"],
            "term": dag.pretty(term)[:60],
            "bound": bound,
            "model": {record["variables"][v]["name"]: val for v, val in assignment.items()},
            "unsettable": targets["unsettable"],
            #: Per unsettable variable, the values the successful runs gave
            #: it: a variable the model pins to a value no run reaches is
            #: where Φ and the program part company.
            "observed": {name: [] for name in targets["unsettable"]},
            "reached": None,
            "runs": 0,
            "failed_runs": 0,
            "verdict": "",
            "note": "",
        }
        unsettable_ids = {var: record["variables"][var]["name"] for var in assignment
                          if record["variables"][var]["name"] in targets["unsettable"]}
        best: int | None = None
        rng = np.random.default_rng(len(rows) + 1)
        settings = draw_inputs.DrawSettings(mode="benign", max_axis=10**9)
        drawers = [
            ModelDrawer(base, targets["symbols"], targets["lengths"], targets["scalars"])
            for base in (draw_inputs.Drawer(domain, rng, settings),
                         wide_drawer(draw_inputs.Drawer)(domain, rng, settings))
        ]
        for attempt in range(1 + draws):
            try:
                values = drawers[attempt % 2].tuple_for(specs)
            except NotImplementedError as exc:
                row["note"] = f"the generator cannot draw this input: {first_line(exc, 120)}"
                break
            trace = evaluate(payloads, values, impls)
            row["runs"] += 1
            if trace.gap is not None:
                row["note"] = f"the reference evaluator cannot run the program: {trace.gap[:120]}"
                break
            if not trace.ok:
                row["failed_runs"] += 1
                continue
            # θ(u) as the adequacy harness reads it, the declared axis of a
            # reshape disambiguated the same way (`adequacy.check`).
            theta, _, ambiguous = theta_of(phase0_record, trace)
            theta = _disambiguate(phase0_record, theta, ambiguous)
            for var, name in unsettable_ids.items():
                seen = row["observed"][name]
                if var in theta and theta[var] not in seen:
                    seen.append(theta[var])
            try:
                value = int(dag.evaluate(term, theta))
            except (UnboundVariable, ValueError, TypeError) as exc:
                row["note"] = f"the term could not be evaluated under θ(u): {first_line(exc, 120)}"
                break
            best = value if best is None else max(best, value)
            if best >= bound:
                break
        row["reached"] = best
        if row["note"]:
            row["verdict"] = "not_run"
        elif best is None:
            row["verdict"] = "no_successful_run"
        elif best > bound:
            row["verdict"] = "unsound"
        elif best == bound:
            row["verdict"] = "tight"
        elif targets["unsettable"]:
            row["verdict"] = "unrealised"
        else:
            row["verdict"] = "loose"
        rows.append(row)
    return rows


def _worker(kernel_path: str, function: str, cache_file: str, draws: int, queue: Any) -> None:
    limit_address_space(ADDRESS_SPACE_LIMIT)
    sys.path.insert(0, str(REPO_ROOT))
    try:
        queue.put({"status": "ok", "rows": _check_circuit(kernel_path, function, cache_file, draws)})
    except BaseException as exc:  # PyO3 panics inherit from BaseException
        queue.put({"status": "harness", "reason": type(exc).__name__,
                   "detail": first_line(exc, 300)})


def has_models(entry: Entry) -> bool:
    return entry.accepted and entry.phase1 is not None and any(
        a["kind"] == "MAX" and a["layer"] == "z3" and a["evidence"]["kind"] == "model"
        for a in entry.phase1["answers"]
    )


def rows(entry: Entry, circuit: Circuit, draws: int = DRAWS) -> list[dict[str, Any]]:
    """The rows of one circuit, computed in a child process."""
    if not has_models(entry):
        return []
    outcome = run_in_child(
        _worker, (str(circuit.kernel_path), circuit.function, str(entry.path), draws),
        ChildLimits(ADDRESS_SPACE_LIMIT, TIMEOUT_SECONDS),
    )
    answer = outcome.answer
    if answer is None or answer.get("status") != "ok":
        detail = (answer or {}).get("detail") or outcome.detail
        return [{"label": entry.label, "answer": None, "term": "", "bound": None, "model": {},
                 "unsettable": [], "observed": {}, "reached": None, "runs": 0, "failed_runs": 0,
                 "verdict": "not_run", "note": f"the child did not answer: {detail}"[:200]}]
    return answer["rows"]


def summary(table: list[dict[str, Any]]) -> dict[str, Any]:
    verdicts = Counter(r["verdict"] for r in table)
    return {
        "maxes": len(table),
        "circuits": len({r["label"] for r in table}),
        "verdicts": dict(verdicts),
        "tight": verdicts.get("tight", 0),
        "loose": verdicts.get("loose", 0),
        "unrealised": verdicts.get("unrealised", 0),
        "unsound": verdicts.get("unsound", 0),
        "not_run": verdicts.get("not_run", 0) + verdicts.get("no_successful_run", 0),
        "loose_rows": [r for r in table if r["verdict"] == "loose"],
        "unsound_rows": [r for r in table if r["verdict"] == "unsound"],
        "unrealised_kinds": Counter(name for r in table if r["verdict"] == "unrealised"
                                    for name in r["unsettable"]),
        "notes": Counter(r["note"][:80] for r in table if r["note"]),
    }


def table(source: Iterable[Entry], draws: int = DRAWS) -> list[dict[str, Any]]:
    circuits = {c.label: c for c in parity_set()}
    out: list[dict[str, Any]] = []
    for entry in source:
        circuit = circuits.get(entry.label)
        if circuit is None:
            continue
        out.extend(rows(entry, circuit, draws))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--only", action="append", default=[])
    arguments = parser.parse_args()
    circuits = parity_set()
    if arguments.only:
        circuits = [c for c in circuits if any(p in c.label for p in arguments.only)]
    source = (Entry.load(file_of(c, CACHE_DIR)) for c in circuits if is_current(file_of(c, CACHE_DIR)))
    result = table(source, arguments.draws)
    for row in result:
        print(f"  {row['verdict']:<18} {row['label']:<44} bound {str(row['bound']):>6} "
              f"reached {str(row['reached']):>6} runs {row['runs']:>2} "
              f"{row['term'][:40]:<40} {row['note'][:60]}")
    totals = summary(result)
    print(f"{totals['maxes']} MAX answers with a model over {totals['circuits']} circuits: "
          f"{totals['tight']} tight, {totals['loose']} loose, {totals['unrealised']} unrealised, "
          f"{totals['unsound']} unsound, {totals['not_run']} not run")
    return 1 if totals["unsound"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
