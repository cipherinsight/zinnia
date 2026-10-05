"""The phase 1 sweep: phase 0 and phase 1 over the circuits of the parity set
a step of the rule library can reach.

Three selections, one flag each.

`--scalar` (the default of step 1) takes the circuits whose Δ holds only
rank-0 values plus arrays that are inputs and are returned or measured by
`len` / `shape`: the nine scalar rules reach exactly those.

`--all` (step 3) takes every circuit of the parity set: the rules of
`revision/phase1/rules/README.md` §11 are the whole library, so a circuit
outside them is a rejection that names a rule or a documented gap rather
than a selection.

`--fixed-shape` (step 2) takes the circuits whose arrays have fixed shapes:
Φ holds no variable of the count, ℓ, c′ or σ kinds, and no phase 0 rule that
produces a `Masked` or a run-time-length `Packed` ran — the filters,
`nonzero`, `unique`, a concatenation with run-time lengths, and
`E-ValueSized`. The 22 rules of step 2 cover that subset; a circuit outside
it is skipped rather than rejected, so that the rejections the report lists
are the library's and not the selection's.

Both run each circuit in its own process and report what phase 1 built and
what the oracle cost.

What the report says per circuit: accepted or rejected with the reason, the
requirement and implementation node counts, the root count, the oracle
statistics of the step-1 brief §3.2.6, and whether the invariant checker of
`revision/phase1/README.md` §2 passed. The summary at the end is the table
the scalability question is answered from: questions per circuit against the
size of Δ and of Φ, the share answered without Z3, how many Z3 queries were
unsat, the largest cone, and the total solver time.

Run it from `zinnia-src`:

```text
MEMRUN_MAX=20G <memrun.sh> env PYTHON=.venv/bin/python \\
  .venv/bin/python -m testing.phase1.sweep --all
```

`--only TEXT` restricts the parity run to the circuits whose label contains
`TEXT`, repeatable, for a check that does not re-measure the whole set.

The report also carries the comparison `revision/README.md` §11 asks for: per
accepted circuit with a capacity decided by a MAX, the bound interval
arithmetic over the ranges alone gives (the trivial bound) beside the MAX
answer (the derived bound), and whether the MAX is tight.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from testing.phase0 import REPO_ROOT
from testing.phase0.corpus import (
    ADDRESS_SPACE_LIMIT,
    ChildLimits,
    Circuit,
    first_line,
    limit_address_space,
    load_module,
    parity_set,
    run_in_child,
)

#: Wall-clock seconds one circuit may take, both phases together.
TIMEOUT_SECONDS = 180.0

#: Milliseconds one Z3 query may take, as `phi_solver.py` sets it.
QUERY_TIMEOUT_MS = 3000

#: Milliseconds of Z3 work one circuit may take, across every query.
PROGRAM_TIMEOUT_MS = 60_000


# ---------------------------------------------------------------------------
# Selection: the circuits step 1's rule library can reach
# ---------------------------------------------------------------------------


def array_leaves(desc: dict[str, Any]) -> int:
    """How many array leaves a descriptor has, composites included."""
    kind = desc.get("kind")
    if kind == "array":
        return 1
    if kind == "composite":
        return sum(array_leaves(element) for element in desc.get("elems", ()))
    return 0


#: The variable kinds a fixed-shape circuit has none of: a count, the
#: declared length of a value-sized construct, `unique`'s count, and the
#: product of two run-time extents. Each of them is a run-time length, which
#: the variable-length rules of step 3 bring.
RUNTIME_LENGTH_KINDS = ("count", "value_sized", "unique_count", "size_product")

#: The phase 0 rules that produce a `Masked` bundle or a `Packed` whose length
#: is a run-time value.
RUNTIME_LENGTH_RULES = (
    "E-Filter-1D", "E-Filter-Axis0", "E-Filter-Same", "E-Nonzero", "E-Unique",
    "E-ValueSized", "E-Comp",
)


def is_fixed_shape_circuit(record: dict[str, Any]) -> tuple[bool, str]:
    """Whether every array of the circuit has a shape fixed before the run.

    The test is on Φ and on the rules Δ recorded, not on the descriptors: a
    run-time length reaches phase 1 either as a variable of one of the kinds
    above or as a rule that packs by a count.
    """
    for variable in record["variables"]:
        kind = variable["origin"]["kind"]
        if kind in RUNTIME_LENGTH_KINDS:
            return False, f"Φ holds a variable of kind {kind}"
    for entry in record["descriptors"]:
        rules = entry["producer"].get("rules") or []
        for rule in rules:
            if rule in RUNTIME_LENGTH_RULES:
                return False, f"a value is produced by {rule}"
    return True, ""


def is_scalar_circuit(record: dict[str, Any]) -> tuple[bool, str]:
    """Whether every array of Δ is an input, or a relabelling of one.

    Step 1 has `INPUT_STORE` for an array parameter and no rule that computes
    an array, so a circuit is in scope exactly when no array value is
    `Applied`: `len(x)`, `x.shape[k]` and returning `x` untouched keep the
    array an input, and everything else does not.
    """
    for entry in record["descriptors"]:
        if not array_leaves(entry["desc"]):
            continue
        producer = entry["producer"]["kind"]
        if producer == "applied":
            rules = entry["producer"].get("rules") or []
            return False, f"an array is computed by {', '.join(rules) or 'a rule'}"
    return True, ""


# ---------------------------------------------------------------------------
# One circuit, in its own process
# ---------------------------------------------------------------------------


def _worker(kernel_path: str, function: str, selection: str, queue) -> None:
    limit_address_space(ADDRESS_SPACE_LIMIT)
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from zinnia import ZKCircuit

        module = load_module(Path(kernel_path))
        circuit = ZKCircuit.from_method(getattr(module, function))
        record = circuit.analyze_shapes(pretty=False)
    except BaseException as exc:  # PyO3 panics inherit from BaseException
        queue.put({"status": "harness", "reason": type(exc).__name__,
                   "detail": first_line(exc, 200)})
        return
    rejection = record.get("rejected")
    if rejection is not None:
        queue.put({"status": "phase0_rejected", "reason": rejection["reason"],
                   "detail": rejection.get("detail", "")})
        return
    if selection == "all":
        selected, why = True, ""
    else:
        select = (
            is_fixed_shape_circuit if selection == "fixed-shape" else is_scalar_circuit
        )
        selected, why = select(record)
    shape = {
        "descriptors": len(record["descriptors"]),
        "variables": len(record["variables"]),
        "constraints": len(record["constraints"]),
        "checks": sum(1 for c in record["constraints"] if c["kind"] == "check"),
        "terms": len(record["terms"]["nodes"]),
    }
    if not selected:
        queue.put({"status": "skipped", "reason": f"not_{selection}", "detail": why,
                   "shape": shape})
        return
    try:
        result = circuit.build_candidates(QUERY_TIMEOUT_MS, PROGRAM_TIMEOUT_MS)
    except BaseException as exc:
        queue.put({"status": "harness", "reason": type(exc).__name__,
                   "detail": first_line(exc, 200), "shape": shape})
        return
    rejection = result.get("rejected")
    if rejection is not None:
        queue.put({"status": "phase1_rejected", "reason": rejection["reason"],
                   "detail": rejection.get("detail", "")[:200], "shape": shape})
        return
    queue.put({
        "status": "ok",
        "shape": shape,
        "requirements": len(result["requirements"]),
        "implementations": len(result["implementations"]),
        "roots": len(result["roots"]),
        "answers": len(result["answers"]),
        "stats": result["stats"],
        "violations": result["violations"],
        "missing": result["missing"][:3],
        "capacities": capacity_rows(result),
    })


def capacity_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    """The capacities a MAX decided, with the trivial bound beside them.

    One row per (rule, variant, MAX answer) over the nodes still in G
    (`revision/README.md` §11): the *derived* bound is the answer's value,
    the *trivial* one is what interval arithmetic over the `var_range`
    constraints alone gives, and `tight` is the oracle's own flag.
    """
    answers = result["answers"]
    rows: dict[tuple, dict[str, Any]] = {}
    for node in result["implementations"]:
        if node["dropped"]:
            continue
        capacities = sorted(
            name
            for name, value in node["parameters"].items()
            if isinstance(value, str) and (name == "B" or name.startswith("B_") or name == "Q")
        )
        for at in node["answers"]:
            answer = answers[at]
            if answer["kind"] != "MAX" or answer["value"] is None:
                continue
            key = (node["rule"], node["variant"], answer["value"], answer["interval_bound"])
            rows[key] = {
                "rule": node["rule"],
                "variant": node["variant"] or "",
                "capacity": ", ".join(
                    f"{name}={node['parameters'][name]}" for name in capacities
                )[:26],
                "derived": answer["value"],
                "trivial": answer["interval_bound"],
                "tight": answer["tight"],
                "layer": answer["layer"],
            }
    return list(rows.values())


def run_one(circuit: Circuit, selection: str) -> dict[str, Any]:
    """Both phases for one circuit, in a child with a memory cap."""
    limits = ChildLimits(ADDRESS_SPACE_LIMIT, TIMEOUT_SECONDS)
    outcome = run_in_child(
        _worker, (str(circuit.kernel_path), circuit.function, selection), limits
    )
    if outcome.answer is not None:
        return outcome.answer
    return {"status": outcome.status, "reason": outcome.status, "detail": outcome.detail}


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


@dataclass
class Totals:
    """What the summary table adds up."""

    accepted: int = 0
    rejected: int = 0
    refused_by_phase0: int = 0
    skipped: int = 0
    failed: int = 0
    invariants_broken: int = 0
    rows: list[dict[str, Any]] = field(default_factory=list)
    reasons: dict[str, int] = field(default_factory=dict)
    #: The library rules a rejection named, by how often.
    missing: dict[str, int] = field(default_factory=dict)
    #: The phase 0 transfer rules a rejection named, by how often.
    uncovered: dict[str, int] = field(default_factory=dict)
    #: Why a circuit was outside the selection, by how often.
    outside: dict[str, int] = field(default_factory=dict)
    #: The capacities a MAX decided, per circuit (`revision/README.md` §11).
    capacities: list[tuple[str, dict[str, Any]]] = field(default_factory=list)


def row_of(circuit: Circuit, answer: dict[str, Any]) -> dict[str, Any]:
    """One accepted circuit's numbers, for the scalability table."""
    stats = answer["stats"]
    solver = stats["solver"]
    kinds = stats["kinds"]
    maximums = kinds.get("MAX", {"asked": 0, "interval": 0, "syntactic": 0, "z3": 0})
    return {
        "label": circuit.label,
        "delta": answer["shape"]["descriptors"],
        "phi": answer["shape"]["constraints"],
        "checks": answer["shape"]["checks"],
        "requirements": answer["requirements"],
        "implementations": answer["implementations"],
        "roots": answer["roots"],
        "questions": stats["questions"],
        "without_solver": stats["without_solver"],
        "queries": solver["queries"],
        "unsat": solver["unsat"],
        "unknown": solver["unknown"],
        "cone_max": int(solver["cone_constraints"]["max"]),
        "cone_mean": solver["cone_constraints"]["mean"],
        "vars_max": int(solver["cone_variables"]["max"]),
        "millis": int(solver["millis"]),
        "max_asked": maximums["asked"],
        "max_interval": maximums["interval"],
        "max_syntactic": maximums["syntactic"],
        "max_z3": maximums["z3"],
        "sweep_queries": solver.get("sweep_queries", 0),
        "sweep_asserted": solver.get("sweep_asserted", 0),
        "coordinates": stats.get("batch_coordinates", 0),
        "constant_coordinates": stats.get("constant_coordinates", 0),
        "invariants": "pass" if not answer["violations"] else "FAIL",
    }


#: The rules of the library, for the histogram of what a rejection named.
LIBRARY = (
    "INPUT_STORE INPUT_SCALAR CONSTANT CONSTANT_SIZED BUILD_DENSE BUILD_MASKED "
    "PACK_BY_KEY PACK_SEGMENTS CONCAT SETITEM MAKE_REF MAKE_REF_RUNTIME REINDEX_REF "
    "ENUM_COORDS APPLY_INDEX_MAP GATHER READ_DIRECT READ_KEY_MUX READ_ROM ROM_BIND "
    "ROM_CLOSE_EXACT MAP_ALIGNED ELEMENTWISE_BATCH BROADCAST_MAP EXPAND_FIBERS "
    "FOLD_GROUPS MATMUL CUMSUM SORT ARGSORT UNIQUE PREFIX_RANK COUNT NONZERO "
    "SCALAR_OP LEN_SHAPE TERM_TO_WIRES SELECT INDEX_NORMALIZE SHAPE_CHECK "
    "ENFORCE_CHECK EXPORT"
).split()


def named_rules(detail: str) -> list[str]:
    """The library rules a rejection message names, if any."""
    after = detail.split("covers with", 1)
    if len(after) < 2:
        return []
    # The clause ends at the semicolon; what follows lists the rules that
    # *were* tried, which is not what was missing.
    clause = after[1].split(";", 1)[0]
    return [rule for rule in LIBRARY if rule in clause]


def named_phase0_rules(detail: str) -> list[str]:
    """The phase 0 transfer rules a rejection message names.

    The message of a row with no provider is "…, which phase 0 produced by
    E-X, E-Y and the library covers with A / B" (`build.rs`, `provenance`),
    so the two halves say which source operation was not implemented and
    which library rules §9 maps it to.
    """
    after = detail.split("produced by", 1)
    if len(after) < 2:
        return []
    clause = after[1].split("and the library covers with", 1)[0]
    return [rule.strip() for rule in clause.split(",") if rule.strip()]


def print_circuit(circuit: Circuit, answer: dict[str, Any]) -> None:
    status = answer["status"]
    if status == "ok":
        row = row_of(circuit, answer)
        print(
            f"  ok       {circuit.label:<52} "
            f"req {row['requirements']:>4} impl {row['implementations']:>4} "
            f"roots {row['roots']:>3} | Q {row['questions']:>4} "
            f"z3 {row['queries']:>3} unsat {row['unsat']:>3} "
            f"cone<={row['cone_max']:>3} {row['millis']:>5} ms | {row['invariants']}"
        )
        if answer["violations"]:
            for violation in answer["violations"][:3]:
                print(f"           violation: {violation}")
        return
    detail = answer.get("detail", "")
    print(f"  {status:<8} {circuit.label:<52} {answer.get('reason','')}: {detail[:160]}")


def summarise(totals: Totals) -> None:
    print()
    print("=" * 118)
    print(
        f"selected {totals.accepted + totals.rejected + totals.failed}, "
        f"accepted {totals.accepted}, rejected by phase 1 {totals.rejected}, "
        f"harness failures {totals.failed}; "
        f"outside the selection {totals.skipped}, "
        f"refused by phase 0 {totals.refused_by_phase0}"
    )
    if totals.invariants_broken:
        print(f"INVARIANTS BROKEN on {totals.invariants_broken} circuits")
    if totals.reasons:
        print("rejection reasons:")
        for reason, count in sorted(totals.reasons.items(), key=lambda kv: -kv[1]):
            print(f"  {count:>3}  {reason}")
    if totals.uncovered:
        print("the phase 0 rules a rejection named, and the library rules §9 "
              "maps them to:")
        for rule, count in sorted(totals.uncovered.items(), key=lambda kv: -kv[1]):
            print(f"  {count:>3}  {rule}")
    if totals.missing:
        print("what phase 1 was missing, by the rule that would provide it:")
        for rule, count in sorted(totals.missing.items(), key=lambda kv: -kv[1]):
            print(f"  {count:>3}  {rule}")
    if totals.outside:
        print("why a circuit was outside the selection:")
        for why, count in sorted(totals.outside.items(), key=lambda kv: -kv[1])[:12]:
            print(f"  {count:>3}  {why}")
    if not totals.rows:
        return
    print()
    header = (
        f"{'circuit':<46}{'|Δ|':>6}{'|Φ|':>6}{'chk':>5}"
        f"{'req':>6}{'impl':>6}{'Q':>6}{'noZ3':>7}"
        f"{'z3':>5}{'unsat':>7}{'cone~':>7}{'cone^':>7}{'vars^':>7}{'ms':>7}"
    )
    print(header)
    print("-" * len(header))
    for row in sorted(totals.rows, key=lambda r: -r["questions"]):
        print(
            f"{row['label'][:45]:<46}{row['delta']:>6}{row['phi']:>6}{row['checks']:>5}"
            f"{row['requirements']:>6}{row['implementations']:>6}{row['questions']:>6}"
            f"{row['without_solver'] * 100:>6.0f}%{row['queries']:>5}{row['unsat']:>7}"
            f"{row['cone_mean']:>7.1f}{row['cone_max']:>7}{row['vars_max']:>7}"
            f"{row['millis']:>7}"
        )
    print("-" * len(header))
    aggregate = {
        "circuits": len(totals.rows),
        "max_asked": sum(r["max_asked"] for r in totals.rows),
        "max_interval": sum(r["max_interval"] for r in totals.rows),
        "coordinates": sum(r["coordinates"] for r in totals.rows),
        "constant_coordinates": sum(r["constant_coordinates"] for r in totals.rows),
        "questions": sum(r["questions"] for r in totals.rows),
        "queries": sum(r["queries"] for r in totals.rows),
        "unsat": sum(r["unsat"] for r in totals.rows),
        "unknown": sum(r["unknown"] for r in totals.rows),
        "millis": sum(r["millis"] for r in totals.rows),
        "cone_max": max(r["cone_max"] for r in totals.rows),
        "vars_max": max(r["vars_max"] for r in totals.rows),
        "requirements": sum(r["requirements"] for r in totals.rows),
        "implementations": sum(r["implementations"] for r in totals.rows),
    }
    share = 0.0
    if aggregate["questions"]:
        share = 1.0 - aggregate["queries"] / aggregate["questions"]
    print(
        f"totals: {aggregate['circuits']} circuits, "
        f"{aggregate['questions']} questions, {aggregate['queries']} Z3 queries "
        f"({share * 100:.0f}% answered without Z3), "
        f"{aggregate['unsat']} unsat, {aggregate['unknown']} unknown, "
        f"largest cone {aggregate['cone_max']} constraints / "
        f"{aggregate['vars_max']} variables, "
        f"{aggregate['millis']} ms of solver time, "
        f"{aggregate['requirements']} requirement and "
        f"{aggregate['implementations']} implementation nodes"
    )
    # The two shares the step-2 brief §6.5 asks for.
    max_share = 0.0
    if aggregate["max_asked"]:
        max_share = aggregate["max_interval"] / aggregate["max_asked"]
    constant_share = 1.0
    if aggregate["coordinates"]:
        constant_share = aggregate["constant_coordinates"] / aggregate["coordinates"]
    print(
        f"CANFAIL sweep: {sum(r['sweep_queries'] for r in totals.rows)} solver "
        f"questions, {sum(r['sweep_asserted'] for r in totals.rows)} constraints "
        f"asserted"
    )
    slow = [row for row in totals.rows if row["millis"] > 10_000]
    if slow:
        print("circuits over 10 s of solver time:")
        for row in sorted(slow, key=lambda r: -r["millis"]):
            print(
                f"  {row['label'][:52]:<53}{row['millis']:>7} ms over "
                f"{row['queries']} queries, largest cone {row['cone_max']}"
            )
    print(
        f"MAX: {aggregate['max_asked']} asked, "
        f"{aggregate['max_interval']} answered by the interval layer alone "
        f"({max_share * 100:.0f}%); "
        f"batch coordinates: {aggregate['coordinates']}, "
        f"{aggregate['constant_coordinates']} constant "
        f"({constant_share * 100:.0f}%)"
    )


def capacities(totals: Totals) -> None:
    """The derived bound against the trivial one (`revision/README.md` §11).

    One line per capacity a MAX decided: the bound interval arithmetic over
    the ranges alone gives, the MAX answer, and whether the MAX is tight.
    """
    if not totals.capacities:
        return
    print()
    header = (
        f"{'circuit':<34}{'rule':<16}{'variant':<13}{'capacity':<20}"
        f"{'trivial':>9}{'derived':>9}{'tight':>7}{'layer':>10}"
    )
    print("capacities decided by a MAX: the trivial bound against the derived one")
    print(header)
    print("-" * len(header))
    for label, row in totals.capacities:
        trivial = row["trivial"] if row["trivial"] is not None else "-"
        print(
            f"{label[:33]:<34}{row['rule'][:15]:<16}{row['variant'][:12]:<13}"
            f"{row['capacity'][:19]:<20}{str(trivial):>9}{str(row['derived']):>9}"
            f"{'yes' if row['tight'] else 'no':>7}{row['layer']:>10}"
        )
    print("-" * len(header))
    same = sum(
        1 for _, row in totals.capacities
        if row["trivial"] is not None and str(row["trivial"]) == str(row["derived"])
    )
    print(
        f"{len(totals.capacities)} capacities over "
        f"{len({label for label, _ in totals.capacities})} circuits; "
        f"{same} equal the trivial bound, "
        f"{len(totals.capacities) - same} are below it"
    )


# ---------------------------------------------------------------------------
# The synthetic family
# ---------------------------------------------------------------------------
#
# The parity set is a NumPy benchmark corpus, so almost every circuit in it
# computes an array and is outside step 1's nine rules. It therefore says
# little about how the oracle scales with the size of a program. These three
# families are scalar programs of growing size, written here so that the
# question count, the cone size and the solver time can be read against |Δ|
# and |Φ| rather than against one accepted circuit.


SYNTHETIC_HEADER = "from zinnia import *\n\n\n@zk_circuit\n"


def synthetic_sources(sizes: tuple[int, ...]) -> list[tuple[str, str]]:
    """The synthetic programs, as (name, source) pairs."""
    programs: list[tuple[str, str]] = []
    for n in sizes:
        # 1. Chained arithmetic on two data scalars: |Δ| grows, Φ gains one
        #    fresh variable per operation and no constraint that relates them.
        body = ["def arith(k: Integer, m: Integer) -> Integer:", "    s = k", "    t = m"]
        for i in range(n):
            body.append(f"    s = s + t * {i + 1}")
            body.append("    t = t + s")
        body.append("    return s")
        programs.append((f"arith-{n}", SYNTHETIC_HEADER + "\n".join(body) + "\n"))

        # 2. Chained shape arithmetic with one assert per step: Φ gains one
        #    check per step, and every later CANFAIL is asked over all of
        #    them.
        body = [
            "def shapes(x: Integer[Array, 'N<=1024']) -> Integer:",
            "    t = len(x)",
        ]
        for i in range(n):
            body.append("    t = t // 2 + 1")
            # One check that can fail, and one the first already implies: the
            # second is what a CANFAIL answers `unsat` on.
            body.append(f"    assert t <= {max(2, 1024 >> (i + 1))}")
            body.append("    assert t <= 1024")
        body.append("    return t")
        programs.append((f"shapes-{n}", SYNTHETIC_HEADER + "\n".join(body) + "\n"))

        # 3. A chain of merges on shape conditions: one SELECT per branch.
        body = [
            "def branches(x: Integer[Array, 'N<=1024'], k: Integer) -> Integer:",
            "    s = len(x)",
        ]
        for i in range(n):
            body.append(f"    if s > {i}:")
            body.append(f"        s = s + {i + 1}")
            body.append("    else:")
            body.append(f"        s = s - {i + 1}")
        body.append("    return s + k")
        programs.append((f"branches-{n}", SYNTHETIC_HEADER + "\n".join(body) + "\n"))
    return programs


def _synthetic_worker(source: str, name: str, queue) -> None:
    limit_address_space(ADDRESS_SPACE_LIMIT)
    sys.path.insert(0, str(REPO_ROOT))
    import importlib.util
    import tempfile

    try:
        from zinnia import ZKCircuit

        directory = Path(tempfile.mkdtemp(prefix="zinnia-phase1-"))
        path = directory / f"{name.replace('-', '_')}.py"
        path.write_text(source)
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[path.stem] = module
        spec.loader.exec_module(module)
        function = next(
            value for key, value in vars(module).items() if not key.startswith("_")
            and key in ("arith", "shapes", "branches")
        )
        circuit = ZKCircuit.from_method(function)
        record = circuit.analyze_shapes(pretty=False)
    except BaseException as exc:
        queue.put({"status": "harness", "reason": type(exc).__name__,
                   "detail": first_line(exc, 200)})
        return
    rejection = record.get("rejected")
    if rejection is not None:
        queue.put({"status": "phase0_rejected", "reason": rejection["reason"],
                   "detail": rejection.get("detail", "")})
        return
    shape = {
        "descriptors": len(record["descriptors"]),
        "variables": len(record["variables"]),
        "constraints": len(record["constraints"]),
        "checks": sum(1 for c in record["constraints"] if c["kind"] == "check"),
        "terms": len(record["terms"]["nodes"]),
    }
    try:
        result = circuit.build_candidates(QUERY_TIMEOUT_MS, PROGRAM_TIMEOUT_MS)
    except BaseException as exc:
        queue.put({"status": "harness", "reason": type(exc).__name__,
                   "detail": first_line(exc, 200), "shape": shape})
        return
    rejection = result.get("rejected")
    if rejection is not None:
        queue.put({"status": "phase1_rejected", "reason": rejection["reason"],
                   "detail": rejection.get("detail", "")[:200], "shape": shape})
        return
    queue.put({
        "status": "ok",
        "shape": shape,
        "requirements": len(result["requirements"]),
        "implementations": len(result["implementations"]),
        "roots": len(result["roots"]),
        "answers": len(result["answers"]),
        "stats": result["stats"],
        "violations": result["violations"],
        "missing": result["missing"][:3],
    })


def run_synthetic(sizes: tuple[int, ...], totals: Totals) -> None:
    """The synthetic family, each program in its own process."""
    limits = ChildLimits(ADDRESS_SPACE_LIMIT, TIMEOUT_SECONDS)
    print()
    print(f"synthetic family: {len(synthetic_sources(sizes))} programs")
    for name, source in synthetic_sources(sizes):
        outcome = run_in_child(_synthetic_worker, (source, name), limits)
        answer = outcome.answer or {
            "status": outcome.status, "reason": outcome.status, "detail": outcome.detail
        }
        synthetic = Circuit("synthetic", name, name, Path(name))
        print_circuit(synthetic, answer)
        if answer["status"] == "ok":
            totals.accepted += 1
            row = row_of(synthetic, answer)
            totals.rows.append(row)
            if row["invariants"] != "pass":
                totals.invariants_broken += 1
        else:
            totals.failed += 1
            key = f"{answer['status']}:{answer.get('reason', '')}"
            totals.reasons[key] = totals.reasons.get(key, 0) + 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after this many parity circuits")
    parser.add_argument("--only", action="append", default=[],
                        help="run only the parity circuits whose label contains "
                             "this text; repeatable")
    parser.add_argument("--json", type=Path, default=None,
                        help="write the per-circuit rows here")
    parser.add_argument("--synthetic", action="store_true",
                        help="also run the synthetic scalar family")
    parser.add_argument("--only-synthetic", action="store_true",
                        help="run the synthetic family and not the parity set")
    parser.add_argument("--sizes", type=str, default="1,2,4,8,16,32",
                        help="the sizes of the synthetic family")
    parser.add_argument("--fixed-shape", dest="selection", action="store_const",
                        const="fixed-shape", default="scalar",
                        help="select the fixed-shape subset (step 2's rules) "
                             "instead of the scalar one (step 1's)")
    parser.add_argument("--all", dest="selection", action="store_const",
                        const="all",
                        help="select every circuit of the parity set (step 3, "
                             "the whole library)")
    arguments = parser.parse_args()

    totals = Totals()
    sizes = tuple(int(size) for size in arguments.sizes.split(",") if size)
    if arguments.only_synthetic:
        run_synthetic(sizes, totals)
        summarise(totals)
        return 1 if totals.invariants_broken or totals.failed else 0

    circuits = parity_set()
    if arguments.only:
        circuits = [
            circuit
            for circuit in circuits
            if any(pattern in circuit.label for pattern in arguments.only)
        ]
    if arguments.limit:
        circuits = circuits[: arguments.limit]
    print(f"parity set: {len(circuits)} circuits, selection {arguments.selection}")
    for circuit in circuits:
        answer = run_one(circuit, arguments.selection)
        status = answer["status"]
        if status == "skipped":
            totals.skipped += 1
            why = answer.get("detail", "")
            totals.outside[why] = totals.outside.get(why, 0) + 1
            continue
        print_circuit(circuit, answer)
        if status == "ok":
            totals.accepted += 1
            row = row_of(circuit, answer)
            totals.rows.append(row)
            for capacity in answer.get("capacities", ()):
                totals.capacities.append((circuit.label, capacity))
            if row["invariants"] != "pass":
                totals.invariants_broken += 1
        elif status == "phase1_rejected":
            totals.rejected += 1
            key = f"{status}:{answer.get('reason','')}"
            totals.reasons[key] = totals.reasons.get(key, 0) + 1
            for rule in named_rules(answer.get("detail", "")):
                totals.missing[rule] = totals.missing.get(rule, 0) + 1
            for rule in named_phase0_rules(answer.get("detail", "")):
                totals.uncovered[rule] = totals.uncovered.get(rule, 0) + 1
        elif status == "phase0_rejected":
            # Phase 0 refused the program, so phase 1 never saw it: not a
            # phase 1 result either way.
            totals.refused_by_phase0 += 1
            key = f"{status}:{answer.get('reason','')}"
            totals.reasons[key] = totals.reasons.get(key, 0) + 1
        else:
            totals.failed += 1
            key = f"{status}:{answer.get('reason','')}"
            totals.reasons[key] = totals.reasons.get(key, 0) + 1
    if arguments.synthetic:
        run_synthetic(sizes, totals)
    summarise(totals)
    capacities(totals)
    if arguments.json:
        arguments.json.write_text(json.dumps(totals.rows, indent=1))
    return 1 if totals.invariants_broken or totals.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
