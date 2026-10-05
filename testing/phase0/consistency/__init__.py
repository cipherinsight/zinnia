"""The comparison of `revision/phase0/acceptance.md` §3.1.

Five relations are checked between phase 0's record and the baseline's, one
per row of §3.1's table, one module per row. Everything here is a pure
function of the two JSON records — `baseline.py` produces the baseline's,
`baseline.phase0_record_isolated` phase 0's — so the comparison can be
unit-tested on hand-written records, and `test_consistency.py` is only the
driver.

| Module | What is in it |
|---|---|
| `records.py` | the report's dataclasses, and the readers over phase 0's record |
| `inputs.py` | (a) input descriptors |
| `arrays.py` | (b) static array shapes and (c) dynamic envelopes |
| `returns.py` | (d) returned shapes and the public payload |
| `checks.py` | (e) the checks of $\\Phi$, and the baseline's facts |

This module holds :func:`compare`, which runs all five, and the three
decisions below, which every one of them rests on.

| §3.1 row | Function | Relation |
|---|---|---|
| input descriptors | :func:`compare_inputs` | equal static shapes; a bounded axis maps to a variable whose range is the baseline's bound |
| static array shapes | :func:`match_static_arrays` | flat sizes, matched in allocation order |
| dynamic envelopes | :func:`compare_dynamic_envelopes` | per axis, $\\max(\\mathrm{dims}[k])$ over $\\Phi\\wedge\\psi$ at most the baseline's bound |
| returned shapes | :func:`check_returns` | the delivery dims at their bound lie in the baseline's `allowed_meta` |
| SMT-resolved facts | :func:`discharge_checks`, :func:`compare_facts` | every check answered with Z3; the baseline's own facts listed |

### Answering a check

Z3 answers the checks phase 0 records, under the terms the export carries.
Three decisions, each made once here:

* **The terms are built from the JSON DAG, not from the exported SMT-LIB
  text.** Every descriptor and constraint of the record addresses a term by
  its *node id*, so a harness that parsed the text would have to rebuild that
  id map anyway, and would inherit the text's `define-fun`-per-node
  indirection. The DAG is one upward sweep
  (:class:`Z3Arena`), and the floor `div` / `mod` rewrites are the ones
  `src/phase0/term/print.rs` states: SMT-LIB's `div` is Euclidean and agrees
  with Python's floor division only for a positive divisor, so a negative
  constant divisor `k` becomes `(div (- t) (- k))`. `wrap64` needs no case:
  `term/mod.rs::wrap64` builds it from `add`, `mod` and `sub`. The text has a second
  problem the DAG does not: a variable named after a composite element, such
  as `a[0]`, is written into the SMT-LIB unquoted, which no parser accepts.
* **A check is answered without assuming itself.** `definition.md` §2 makes
  every check a constraint of $\\Phi$, so asking "can this `defcond` be false
  under $\\Phi$" with the full $\\Phi$ is vacuous: the constraint forbids it.
  The harness answers under the *facts* of $\\Phi$ — the variable ranges, the
  `scalar_requires` preconditions and the rules' declarations — which is what
  README 6.1's acyclicity condition allows a circuit to rely on. The MAX
  queries of the dynamic envelopes use the full $\\Phi$, checks included,
  which is what §2 prescribes for a phase 1 question.
* **The product facts of §3.4 are asserted, and dropped on `unknown`.**
  $\\sigma=\\prod_k\\mathrm{dims}[k]$ is nonlinear and therefore absent from
  the SMT-LIB text, but it is a true fact and it is what makes a reshape
  bound finite. Every query runs first with the products and, if the answer
  is `unknown`, once more without them; the row records which answer it is.

### What cannot be extracted from the baseline

§3.1's last row asks for "every fact the baseline needed". The IR does not
carry them. `visit_while`'s guard resolution, `resolve_shape_type`'s
exact/upper SMT queries and the constant folding of an index all run inside
`IRGenerator`, and their answers are consumed there: a guard proved false
shortens the statement list, a reshape extent proved constant becomes a
number in a shape, an index proved in range emits nothing. What survives in
the IR is the complement — the checks the baseline could **not** discharge,
which it emitted as `Assert*` statements. :func:`compare_facts` therefore
reports the enforced checks against phase 0's and names the discharged facts
as unextractable, rather than approximating them. The one exception is a
symbolic input axis the baseline resolved *exactly*: that answer survives in
`program_inputs[i].export()["dt"]` as a concrete
`NDArray` shape where the annotation named a symbol, and
:func:`compare_inputs` checks it.
"""

from __future__ import annotations

import time
from typing import Any

from testing.phase0.baseline import BaselineRecord
from testing.phase0.consistency.arrays import (
    compare_dynamic_envelopes,
    match_static_arrays,
)
from testing.phase0.consistency.checks import compare_facts, discharge_checks
from testing.phase0.consistency.inputs import compare_inputs
from testing.phase0.consistency.records import (
    ArrayEntry,
    AxisBound,
    CheckResult,
    CircuitConsistency,
    DynamicMatch,
    FactSummary,
    InputCheck,
    StaticMatch,
    ReturnCheck,
    allocated_arrays,
    cut,
    descriptor_leaves,
    input_descriptors,
)
from testing.phase0.consistency.returns import check_return_payload, check_returns
from testing.phase0.phi_solver import CIRCUIT_BUDGET_SECONDS, MAX_ARENA_NODES, PhiSolver
from testing.phase0.terms import TermDag

__all__ = [
    "ArrayEntry",
    "AxisBound",
    "CheckResult",
    "CircuitConsistency",
    "DynamicMatch",
    "FactSummary",
    "InputCheck",
    "PhiSolver",
    "ReturnCheck",
    "StaticMatch",
    "TermDag",
    "allocated_arrays",
    "check_return_payload",
    "check_returns",
    "compare",
    "compare_dynamic_envelopes",
    "compare_facts",
    "compare_inputs",
    "cut",
    "descriptor_leaves",
    "discharge_checks",
    "input_descriptors",
    "match_static_arrays",
]


def compare(
    phase0: dict[str, Any],
    baseline: BaselineRecord,
    budget_seconds: float = CIRCUIT_BUDGET_SECONDS,
) -> CircuitConsistency:
    """Every relation of §3.1, for one circuit whose two records are in hand."""
    row = CircuitConsistency(baseline.program, baseline.function, "ok")
    dag = TermDag.from_json(phase0["terms"])
    started = time.time()
    solver = _make_solver(phase0, budget_seconds, row)
    row.seconds["build"] = time.time() - started
    row.sizes = {
        "variables": len(phase0["variables"]),
        "constraints": len(phase0["constraints"]),
        "terms": len(phase0["terms"]["nodes"]),
        "descriptors": len(phase0["descriptors"]),
        "baseline_ir_statements": baseline.ir_statements,
    }

    at = time.time()
    row.inputs = compare_inputs(phase0, baseline, dag, solver)
    entries = allocated_arrays(phase0, dag)
    row.static_arrays = match_static_arrays(entries, baseline)
    row.dynamic_arrays = compare_dynamic_envelopes(entries, baseline, dag, solver)
    delivered = descriptor_leaves(phase0, dag, delivery=True)
    row.returns = check_returns(
        [entry for entry in delivered if entry.kind == "array"], baseline, solver
    )
    row.returns.append(check_return_payload(delivered, baseline, solver))
    row.seconds["shapes"] = time.time() - at

    at = time.time()
    row.checks = discharge_checks(phase0, solver)
    row.facts = compare_facts(phase0, baseline)
    row.seconds["checks"] = time.time() - at
    row.seconds["z3"] = solver.spent_seconds if solver else 0.0
    return row


def _make_solver(
    phase0: dict[str, Any], budget_seconds: float, row: CircuitConsistency
) -> PhiSolver | None:
    """$\\Phi$ in Z3, unless the arena is too large to be worth building."""
    nodes = len(phase0["terms"]["nodes"])
    if nodes > MAX_ARENA_NODES:
        row.detail = (
            f"the Z3 stage was skipped: the term arena has {nodes} nodes, past "
            f"the harness limit of {MAX_ARENA_NODES}"
        )
        return None
    return PhiSolver(phase0, budget_seconds)
