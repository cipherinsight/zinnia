"""(a) Input descriptors: `acceptance.md` §3.1, row 1.

Phase 0's input descriptor and the baseline's resolved `program_inputs[i]`
must agree: equal static shapes, and a bounded axis of the annotation mapped
to a variable of $\\Phi$ whose range is the baseline's bound. A symbolic axis
the baseline resolved *exactly* is the one fact of §3.1's last row that does
survive into the IR, and it is checked here.
"""

from __future__ import annotations

from typing import Any

from testing.phase0.baseline import BaselineRecord
from testing.phase0.consistency.records import (
    AxisBound,
    InputCheck,
    axis_by_solver,
    cut,
    input_descriptors,
)
from testing.phase0.phi_solver import PhiSolver
from testing.phase0.terms import TermDag


def describe_dt(dt: Any) -> str:
    """A one-line rendering of a baseline input type."""
    if isinstance(dt, str):
        return dt
    if not isinstance(dt, dict) or len(dt) != 1:
        return repr(dt)
    (variant, body), = dt.items()
    if variant == "NDArray":
        return f"NDArray{body.get('shape', [])}:{body.get('dtype')}"
    if variant == "DynamicNDArray":
        return (
            f"DynamicNDArray{body.get('max_shape', [])}"
            f"(len<={body.get('max_length')}):{body.get('dtype')}"
        )
    if variant == "SymbolicNDArray":
        return f"SymbolicNDArray{body.get('shape', [])}:{body.get('dtype')}"
    if variant in ("List", "Tuple"):
        return f"{variant}[{', '.join(describe_dt(e) for e in body.get('elements', []))}]"
    return variant


#: The phase 0 descriptor kind a baseline scalar type maps to (§3.2, §3.3).
_SCALAR_KIND = {"Class": "class", "None": "none"}


def baseline_envelope(dt: Any) -> tuple[str, list[int] | None]:
    """`(kind, per-axis bound)` for a baseline input type.

    The kind is `static_array` (the bound is the exact shape),
    `dynamic_array` (the bound is `max_shape`), a phase 0 descriptor kind for
    a scalar, `composite`, or `other` for a type the harness does not
    compare — a `PoseidonHashed` wrapper, whose phase 0 descriptor is the
    inner type's and whose shape the baseline states nowhere.
    """
    if isinstance(dt, str):
        return _SCALAR_KIND.get(dt, "scalar"), None
    if not isinstance(dt, dict) or len(dt) != 1:
        return "other", None
    (variant, body), = dt.items()
    if variant == "NDArray":
        return "static_array", [int(d) for d in body.get("shape", [])]
    if variant == "DynamicNDArray":
        shape = [int(d) for d in body.get("max_shape", [])]
        return "dynamic_array", shape or [int(body.get("max_length", 0))]
    if variant in ("List", "Tuple"):
        return "composite", None
    return "other", None


def pinned_value(record: dict[str, Any], dag: TermDag, node_id: int) -> int | None:
    """The single value $\\Phi$ pins a dim to, syntactically, or `None`.

    Either a literal extent or a variable whose range has equal constant
    bounds — the form `coverage.md` §4 gives an input axis the baseline
    resolved *exactly* ("an exact binding is a constant").
    """
    node = dag.nodes[int(node_id)]
    if node["op"] == "const":
        return int(node["value"])
    if node["op"] != "var":
        return None
    info = record["variables"][int(node["var"])]
    if info["lo"] is None or info["hi"] is None:
        return None
    low, high = dag.nodes[int(info["lo"])], dag.nodes[int(info["hi"])]
    if low["op"] != "const" or high["op"] != "const" or low["value"] != high["value"]:
        return None
    return int(low["value"])


def _axis_from_range(
    record: dict[str, Any], dag: TermDag, node_id: int, bound: int, axis: int
) -> AxisBound | None:
    """The axis verdict from the syntactic range of $\\Phi$, when there is one.

    §3.1's relation for a bounded input axis is "maps to a variable whose
    range upper bound equals the baseline's bound", which is a lookup and not
    a solver call; a dim that is not a bare variable falls through to Z3.
    """
    node = dag.nodes[int(node_id)]
    if node["op"] == "const":
        value = int(node["value"])
    elif node["op"] == "var":
        info = record["variables"][int(node["var"])]
        if info["hi"] is None:
            return None
        hi = dag.nodes[int(info["hi"])]
        if hi["op"] != "const":
            return None
        value = int(hi["value"])
    else:
        return None
    verdict = "equal" if value == bound else ("tighter" if value < bound else "looser")
    return AxisBound(
        axis=axis,
        baseline=bound,
        phase0=value,
        expression=cut(dag.pretty(int(node_id))),
        verdict=verdict,
    )


def _axis_bounds(
    record: dict[str, Any],
    dag: TermDag,
    solver: PhiSolver | None,
    entry_dims: list[int],
    psi: int,
    bounds: list[int],
) -> list[AxisBound]:
    """One :class:`AxisBound` per axis, by range lookup then by Z3."""
    axes = []
    for axis, (node_id, bound) in enumerate(zip(entry_dims, bounds)):
        syntactic = _axis_from_range(record, dag, node_id, bound, axis)
        if syntactic is not None:
            axes.append(syntactic)
            continue
        axes.append(axis_by_solver(dag, solver, node_id, psi, bound, axis))
    return axes


def compare_inputs(
    phase0: dict[str, Any],
    baseline: BaselineRecord,
    dag: TermDag,
    solver: PhiSolver | None,
) -> list[InputCheck]:
    """§3.1, row 1: the input descriptors of the two compilers."""
    descriptors = input_descriptors(phase0)
    checks = []
    for parameter in baseline.inputs:
        desc = descriptors.get(parameter.name)
        rendered = describe_dt(parameter.resolved)
        if desc is None:
            checks.append(
                InputCheck(
                    parameter.name,
                    rendered,
                    "—",
                    "missing_in_phase0",
                    detail="phase 0 recorded no descriptor for this parameter",
                )
            )
            continue
        checks.append(_compare_one_input(phase0, dag, solver, parameter, desc, rendered))
    return checks


def _compare_one_input(
    phase0: dict[str, Any],
    dag: TermDag,
    solver: PhiSolver | None,
    parameter: Any,
    desc: dict[str, Any],
    rendered: str,
) -> InputCheck:
    """One parameter's verdict; see :func:`compare_inputs`."""
    kind, bounds = baseline_envelope(parameter.resolved)
    name = parameter.name
    if kind == "other":
        return InputCheck(
            name, rendered, desc["kind"], "not_compared",
            detail="the baseline states no shape for this type",
        )
    if bounds is None:
        verdict = "equal" if desc["kind"] == kind else "kind_differs"
        detail = None if verdict == "equal" else f"phase 0 has a {desc['kind']}"
        return InputCheck(name, rendered, desc["kind"], verdict, detail=detail)
    if desc["kind"] != "array":
        return InputCheck(
            name, rendered, desc["kind"], "kind_differs",
            detail=f"the baseline has an array, phase 0 a {desc['kind']}",
        )
    dims = [int(d) for d in desc["dims"]]
    phase0_text = f"Array[{', '.join(desc['dims_pretty'])}]"
    if len(dims) != len(bounds):
        return InputCheck(
            name, rendered, phase0_text, "rank_differs",
            detail=f"the baseline has rank {len(bounds)}, phase 0 rank {len(dims)}",
        )
    if kind == "static_array":
        return _compare_static_input(phase0, dag, name, rendered, phase0_text, dims, bounds)
    axes = _axis_bounds(phase0, dag, solver, dims, int(desc["psi"]), bounds)
    verdict = "looser" if any(a.fails for a in axes) else (
        "tighter" if any(a.verdict == "tighter" for a in axes) else "equal"
    )
    return InputCheck(name, rendered, phase0_text, verdict, axes)


def _compare_static_input(
    record: dict[str, Any],
    dag: TermDag,
    name: str,
    rendered: str,
    phase0_text: str,
    dims: list[int],
    shape: list[int],
) -> InputCheck:
    """A parameter whose shape the baseline pinned: the shapes must be equal.

    "Equal" is exact on both sides: the baseline states a literal extent, and
    phase 0 must pin the axis to the same value — as a literal, or as a
    variable whose range has equal bounds, which is how an exactly resolved
    shape symbol arrives (`coverage.md` §4).
    """
    differing = []
    for axis, (node_id, extent) in enumerate(zip(dims, shape)):
        value = pinned_value(record, dag, node_id)
        if value != extent:
            differing.append(
                f"axis {axis}: the baseline says {extent}, phase 0 "
                f"{cut(dag.pretty(int(node_id)))}"
            )
    return InputCheck(
        name, rendered, phase0_text,
        "equal" if not differing else "shape_differs",
        detail=None if not differing else "; ".join(differing),
    )
