"""Inputs A and B for the acceptance study (`fixture.json` per entry function).

Same schema as the performance set (datasets/ds1000/*/fixture.json), with two
additions: `"kind": "tuple"` parameters carry `"items"` (one description per
element) and their value is the list of element values; complex values are
[real, imag] pairs.

* NPBench: A is NPBench's own `initialize()` data at preset T; B is a seeded
  perturbation of A.
* Pythran: A is the (scaled) example call's arguments; B is a seeded
  perturbation of A.
* DS-1000: A and B come from the performance set's draw for the problem when
  there is one (the shape parameters the performance kernel binds as constants
  take their DS-1000 value); otherwise A is DS-1000's own last test input and B
  a seeded perturbation of it.

The perturbation keeps sizes, trip counts and every other integer scalar, and
tries in order: (1) float values plus Gaussian noise of 10 % of their largest
magnitude, integer arrays redrawn uniformly within their range, booleans
flipped with probability 1/4; (2) the same with multiplicative float noise
(signs and zeros kept); (3) and (4) the same with integer arrays kept (they are
often indices); (5) every float multiplied by one random factor in [1.05, 1.25],
which keeps signs, order and positive definiteness (Cholesky inputs). A kernel
with only integer scalars gets each of them minus one (plus one for 0 and 1),
which keeps sizes at the cap within it.
The first B on which the kernel runs is kept.

Check, for A and B: the kernel body runs as plain NumPy without an error
(division by zero, overflow and invalid operations raise) and every numeric
output, and every argument after the call (kernels that update an input in
place and return None), is finite. Whether Zinnia compiles the entry is the
acceptance measurement itself and is not checked here. The notes flag values
Q32 cannot hold: inputs or outputs of magnitude 2^31 or more, and nonzero
inputs below the Q32 resolution 2^-32 (they become 0).
"""

from __future__ import annotations

import ast
import copy
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy_reference  # noqa: E402
from perf_fixtures import format_fixture  # noqa: E402

Q32_LIMIT = 2.0 ** 31
Q32_RESOLUTION = 2.0 ** -32
JSON_DTYPE = {"Float": "float", "Integer": "int", "Boolean": "bool", "Complex": "complex"}


class FixtureError(Exception):
    pass


# ---------------------------------------------------------------------------
# Parameter descriptions from annotations
# ---------------------------------------------------------------------------


def describe(annotation: str) -> dict[str, Any]:
    """`Float[Array, "4 4"]` -> {"kind": "array", "dtype": "float", "shape": [4, 4]}."""
    node = ast.parse(annotation, mode="eval").body
    return _describe(node)


def _describe(node: ast.expr) -> dict[str, Any]:
    if isinstance(node, ast.Name):
        return {"kind": "scalar", "dtype": JSON_DTYPE[node.id]}
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        if node.value.id == "Tuple":
            items = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            return {"kind": "tuple", "items": [_describe(item) for item in items]}
        dtype = JSON_DTYPE[node.value.id]
        shape_text = node.slice.elts[1].value  # type: ignore[attr-defined]
        return {"kind": "array", "dtype": dtype, "shape": [int(x) for x in shape_text.split()]}
    raise FixtureError(f"cannot describe annotation {ast.unparse(node)}")


def coerce(value: Any, desc: dict[str, Any]) -> Any:
    """The example value as the kernel's parameter type expects it."""
    if desc["kind"] == "tuple":
        return tuple(coerce(v, d) for v, d in zip(value, desc["items"]))
    numpy_dtype = {"float": np.float64, "int": np.int64, "bool": np.bool_, "complex": np.complex128}[desc["dtype"]]
    if desc["dtype"] == "complex":                                   # fixture.json [real, imag] pairs
        pairs = np.asarray(value)
        if pairs.dtype.kind in "iuf" and pairs.shape[-1:] == (2,) and list(pairs.shape[:-1]) == desc.get("shape", []):
            value = pairs[..., 0] + 1j * pairs[..., 1]
    array = np.asarray(value, dtype=numpy_dtype)
    if desc["kind"] == "array":
        if list(array.shape) != desc["shape"]:
            raise FixtureError(f"value of shape {array.shape}, annotation says {desc['shape']}")
        return array
    return array.item()


def to_json(value: Any, desc: dict[str, Any]) -> Any:
    if desc["kind"] == "tuple":
        return [to_json(v, d) for v, d in zip(value, desc["items"])]
    array = np.asarray(value)
    if desc["dtype"] == "complex":
        return np.stack([array.real, array.imag], axis=-1).astype(float).tolist()
    if desc["dtype"] == "float":
        return array.astype(float).tolist()
    if desc["dtype"] == "bool":
        return array.astype(bool).tolist()
    return array.astype(np.int64).tolist()


# ---------------------------------------------------------------------------
# Perturbation and checks
# ---------------------------------------------------------------------------


def _perturb(value: Any, desc: dict[str, Any], rs: np.random.RandomState, mode: str,
             redraw_ints: bool, factor: float) -> Any:
    if desc["kind"] == "tuple":
        return tuple(_perturb(v, d, rs, mode, redraw_ints, factor) for v, d in zip(value, desc["items"]))
    array = np.asarray(value)
    if desc["dtype"] in ("float", "complex"):
        def noise(part: np.ndarray) -> np.ndarray:
            if mode == "scale":
                return part * factor
            if mode == "multiplicative":
                return part * (1 + 0.05 * rs.standard_normal(part.shape))
            scale = 0.1 * (float(np.max(np.abs(part))) if part.size and np.max(np.abs(part)) > 0 else 1.0)
            return part + scale * rs.standard_normal(part.shape)
        changed = noise(array.real.astype(float))
        if desc["dtype"] == "complex":
            changed = changed + 1j * noise(array.imag.astype(float))
        return changed if desc["kind"] == "array" else changed.item()
    if desc["dtype"] == "bool" and desc["kind"] == "array":
        return np.logical_xor(array, rs.rand(*array.shape) < 0.25)
    if desc["dtype"] == "int" and desc["kind"] == "array" and redraw_ints:
        low, high = int(array.min()), int(array.max())
        return rs.randint(low, max(high, low + 1) + 1, array.shape)
    return value


def same(a: dict[str, Any], b: dict[str, Any]) -> bool:
    def flat(v: Any) -> list:
        if isinstance(v, tuple):
            return [x for item in v for x in flat(item)]
        return list(np.asarray(v).ravel())
    return all(len(flat(a[k])) == len(flat(b[k])) and all(x == y for x, y in zip(flat(a[k]), flat(b[k]))) for k in a)


def numbers(value: Any) -> list[complex]:
    """Every numeric leaf of a kernel output (None and other objects are skipped)."""
    if isinstance(value, (list, tuple)):
        return [x for item in value for x in numbers(item)]
    if value is None or isinstance(value, (str, bytes)):
        return []
    try:
        array = np.asarray(value)
    except Exception:  # noqa: BLE001
        return []
    if array.dtype.kind not in "biufc":
        return []
    return [complex(x) for x in array.ravel()]


def run(kernel: Path, function: str, args: list[Any]) -> tuple[list[complex], str | None]:
    """Every number the call produces (its output, then its arguments after the
    call) and a problem description (None when fine). `args` is not changed, and
    each argument is copied on its own: Zinnia's parameters never alias, even
    when the example passes one array twice (d2q9's one_time_step(r, r))."""
    copies = [copy.deepcopy(arg) for arg in args]
    try:
        output = numpy_reference.run_kernel_in_place(kernel, copies, function)
    except Exception as exc:  # noqa: BLE001
        return [], f"{type(exc).__name__}: {exc}"
    values = numbers(output) + numbers(copies)
    if not all(math.isfinite(v.real) and math.isfinite(v.imag) for v in values):
        return values, "non-finite output"
    return values, None


def draw_b(kernel: Path, function: str, order: list[str], descs: dict[str, dict[str, Any]],
           a: dict[str, Any], seed: int) -> tuple[dict[str, Any], str]:
    attempts = [("additive", True, "noise 10 % of the largest magnitude; integer arrays redrawn in their range"),
                ("multiplicative", True, "multiplicative noise 5 %; integer arrays redrawn in their range"),
                ("additive", False, "noise 10 % of the largest magnitude; integer arrays kept"),
                ("multiplicative", False, "multiplicative noise 5 %; integer arrays kept"),
                ("scale", False, "every float value multiplied by one random factor in [1.05, 1.25] (keeps "
                                 "signs, order and positive definiteness); integer arrays kept")]
    for index, (mode, redraw, how) in enumerate(attempts):
        rs = np.random.RandomState(seed + index)
        factor = 1.05 + 0.2 * rs.rand()
        b = {n: coerce(_perturb(a[n], descs[n], rs, mode, redraw, factor), descs[n]) for n in order}
        if not same(a, b) and run(kernel, function, [b[n] for n in order])[1] is None:
            return b, how
    if order and all(descs[n] == {"kind": "scalar", "dtype": "int"} for n in order):
        b = {n: a[n] - 1 if a[n] > 1 else a[n] + 1 for n in order}
        if run(kernel, function, [b[n] for n in order])[1] is None:
            return b, "every integer scalar minus one (plus one for 0 and 1)"
    raise FixtureError("no perturbation of A runs without an error")


def build(case: str, function: str, kernel: Path, annotations: dict[str, str], a_values: dict[str, Any],
          seed: int, source_note: str, b_values: dict[str, Any] | None = None) -> dict[str, Any]:
    """The fixture document; raises FixtureError when A (or B) does not run."""
    order = [n for n in annotations]
    descs = {n: describe(annotations[n]) for n in order}
    a = {n: coerce(a_values[n], descs[n]) for n in order}
    produced, problem = run(kernel, function, [a[n] for n in order])
    if problem:
        raise FixtureError(f"input A: {problem}")
    notes = [source_note]
    if b_values is None:
        if not order:
            b, how = {}, "the kernel has no inputs, so A and B are both empty"
        else:
            b, how = draw_b(kernel, function, order, descs, a, seed)
        notes.append(f"B is a seeded perturbation of A (numpy RandomState({seed})): {how}.")
    else:
        b = {n: coerce(b_values[n], descs[n]) for n in order}
        _, problem = run(kernel, function, [b[n] for n in order])
        if problem:
            raise FixtureError(f"input B: {problem}")
    produced += run(kernel, function, [b[n] for n in order])[0]
    large = [n for n in order for v in (a[n], b[n]) for x in numbers(v) if abs(x) >= Q32_LIMIT]
    if large:
        notes.append(f"Input values of magnitude 2^31 or more in {sorted(set(large))}: outside the Q32 range.")
    tiny = [n for n in order if descs_dtype(descs[n]) != "int"
            for v in (a[n], b[n]) for x in numbers(v) if 0 < abs(x) < Q32_RESOLUTION]
    if tiny:
        notes.append(f"Nonzero input values below 2^-32 in {sorted(set(tiny))}: Q32 rounds them to 0.")
    if any(abs(x) >= Q32_LIMIT for x in produced):
        notes.append("The NumPy outputs reach magnitude 2^31 or more: outside the Q32 range.")
    parameters = []
    for name in order:
        parameters.append({"name": name, **descs[name]})
    return {
        "case": case,
        "function": function,
        "parameters": parameters,
        "constants": {},
        "inputs": {"A": {n: to_json(a[n], descs[n]) for n in order},
                   "B": {n: to_json(b[n], descs[n]) for n in order}},
        "notes": " ".join(notes),
    }


def descs_dtype(desc: dict[str, Any]) -> str:
    """The dtype of a parameter; "mixed" for a tuple of different dtypes."""
    if desc["kind"] == "tuple":
        kinds = {descs_dtype(d) for d in desc["items"]}
        return kinds.pop() if len(kinds) == 1 else "mixed"
    return desc["dtype"]


def fixture_name(functions_in_program: int, function: str) -> str:
    return "fixture.json" if functions_in_program == 1 else f"fixture.{function}.json"


def seed_for(case: str, function: str) -> int:
    """A seed that depends only on the case and function names."""
    digits = int(re.sub(r"\D", "", case) or 0)
    return (sum(map(ord, case + "." + function)) * 1009 + digits) % (2 ** 31)


def dump(path: Path, data: dict[str, Any]) -> None:
    path.write_text(format_fixture(data), encoding="utf-8")

