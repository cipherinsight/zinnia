"""Inputs A and B for the acceptance study (`fixture.json` per entry function).

Same schema as the performance set (datasets/ds1000/*/fixture.json), with two
additions: `"kind": "tuple"` parameters carry `"items"` (one description per
element) and their value is the list of element values; complex values are
[real, imag] pairs.

* NPBench: A is NPBench's own `initialize()` data at preset T; B is a seeded
  perturbation of A.
* Pythran: A is the (scaled) example call's arguments; B is a seeded
  perturbation of A.
* Inputs changed for Q32 (PYTHRAN_Q32_INPUTS, NPBENCH_Q32_INPUTS): where the
  example input puts a program where no Q32 implementation can follow NumPy
  (an exact zero pivot, exp underflowing to 0 before a log, acos at exactly 1,
  a near-zero divisor, outputs beyond 2^31), A (and for lu also B) is changed
  and the reason goes into the fixture notes and the entry's `sizing`.
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

Output tolerance: every acceptance entry has `float_tolerance`; a float output
passes when |actual - expected| <= atol + rtol * |expected| against NumPy on the
same input. The default is atol = rtol = 1e-6 (as for the performance set's
computed outputs). FLOAT_TOLERANCE_OVERRIDES raises atol where the NumPy
program itself is not reproducible in Q32, with the measurement that justifies
it (recomputed when the fixtures are written; the tool stops if it exceeds atol).
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


# ---------------------------------------------------------------------------
# Inputs changed for Q32, and output tolerances
# ---------------------------------------------------------------------------

# (program, function) -> (example line for A, example line for B or None (B is
# then the seeded perturbation of A), why).
PYTHRAN_Q32_INPUTS: dict[tuple[str, str], tuple[str, str | None, str]] = {
    ("pythran_log_likelihood", "log_likelihood"): (
        "import numpy as np ; N = 16 ; rs = np.random.RandomState(16) ; b = 0.1 ; c = 1.1 ; "
        "a = b + c * rs.uniform(-2., 2., N) ; log_likelihood(a, b, c)",
        None,
        "the data are drawn (seeded) within 2 sigma of the mean instead of np.arange(16.), which reaches about "
        "13.5 sigma, where exp(-(x-mean)^2 / 2 sigma^2) is about 1e-40, 0 in Q32, and log(0) is undefined"),
    ("pythran_lu", "factorMatrix0"): (
        "import numpy as np ; rs = np.random.RandomState(50) ; x = rs.uniform(-1., 1., (5, 5)) ; "
        "x += np.diag(np.abs(x).sum(axis=1) + 1.) ; factorMatrix0(x)",
        "import numpy as np ; rs = np.random.RandomState(51) ; x = rs.uniform(-1., 1., (5, 5)) ; "
        "x += np.diag(np.abs(x).sum(axis=1) + 1.) ; factorMatrix0(x)",
        "A and B are seeded strictly diagonally dominant 5x5 matrices (each diagonal entry exceeds the rest of "
        "its row by at least 1, so every pivot stays away from 0) instead of upstream's np.arange(1., 26.)"
        ".reshape(5, 5), which is singular (rank 2) and gives an exact zero pivot in Q32"),
    ("pythran_lu", "factorMatrix1"): (
        "import numpy as np ; rs = np.random.RandomState(52) ; x = rs.uniform(-1., 1., (5, 5)) ; "
        "x += np.diag(np.abs(x).sum(axis=1) + 1.) ; factorMatrix1(x)",
        "import numpy as np ; rs = np.random.RandomState(53) ; x = rs.uniform(-1., 1., (5, 5)) ; "
        "x += np.diag(np.abs(x).sum(axis=1) + 1.) ; factorMatrix1(x)",
        "A and B are seeded strictly diagonally dominant 5x5 matrices of their own (each diagonal entry exceeds "
        "the rest of its row by at least 1) instead of upstream's input, which is factorMatrix0's in-place "
        "output on the singular np.arange(1., 26.).reshape(5, 5): pivots of rounding residue (1e-15), 0 in Q32"),
    ("pythran_hyantes_core", "run"): (
        "run(0.,0.,90.,90., 1., 100., 16, 16, [ [i/10. + 0.007, i/10. + 0.007, i/20.] for i in range(16) ])",
        None,
        "every town is moved by 0.007 rad in both coordinates. In the scaled upstream line the towns at (0, 0) "
        "and (1, 1) sit exactly on grid points, where acos(1 - eps) turns the ~1e-9 error of Q32 trigonometry "
        "into 0.28 km; now they are 51 and 63 km from the nearest grid point (still within range_ = 100 km), "
        "where acos amplifies that error by at most about 1 / sin(0.008) = 125"),
    ("pythran_calculate_u", "timeloop"): (
        "A=[list(range(10)) for i in range(5)] ; B=[list(range(10)) for i in range(5)] ; "
        "C=[list(range(10)) for i in range(5)] ; timeloop(1.,1.45,.1,.1,.2, A,B,C )",
        None,
        "t_stop 2.0 -> 1.45: the trip count 10 -> 5 (the sizing rule allows any trip count up to 16). The "
        "explicit scheme is unstable for these coefficients and grows about 30x per step: 10 steps reach "
        "2.3e13, beyond Q32's 2^31, 5 steps reach 1.1e6. 1.45 rather than 1.5 because the loop runs while "
        "t <= t_stop with t accumulating 0.1: at 1.5 the last comparison is 1.5000000000000004 <= 1.5 in "
        "float64 and depends on whether Q32 rounds or truncates 0.1; at 1.45 the margin is 0.05"),
}

# NPBench short_name -> ({initialize() output: factor applied at preset T}, why).
NPBENCH_Q32_INPUTS: dict[str, tuple[dict[str, float], str]] = {
    "vadv": ({"wcon": 0.1},
             "wcon is NPBench's draw times 0.1. With wcon uniform in [0, 1) and dtr_stage = 0.15, the Thomas "
             "sweep divides by bcol - ccol[k-1] * acol, which reaches 3.7e-5 on A (1.1e-4 on B) and amplifies "
             "Q32 rounding to 1.1e-3. With |wcon| < 0.15 the tridiagonal system is strictly diagonally dominant "
             "(|bcol| > |acol| + |ccol|) whatever the signs, and every divisor stays above 0.1; the "
             "computation is unchanged"),
}

FLOAT_TOLERANCE_DEFAULT = {"atol": 1e-6, "rtol": 1e-6}

# (program, function) -> tolerance, why, and how to measure the effect it covers:
# "q32_inputs": NumPy on the inputs rounded to Q32 against NumPy on the inputs;
# "float64_buffers": NumPy with `dtype=np.float32` buffers made float64 against as written.
FLOAT_TOLERANCE_OVERRIDES: dict[tuple[str, str], dict[str, Any]] = {
    ("npbench_nbody_nbody", "nbody"): {
        "atol": 2e-3, "rtol": 1e-6, "measure": "q32_inputs",
        "reason": "the 16-step n-body integration is chaotic: rounding only the inputs to Q32 moves NumPy's own "
                  "float64 output by up to 1e-3, so no Q32 implementation can be closer; atol is twice that"},
    ("npbench_deep_learning_resnet_resnet", "resnet_basicblock"): {
        "atol": 2e-6, "rtol": 1e-6, "measure": "float64_buffers",
        "reason": "conv2d writes its result into a float32 buffer (np.empty(..., dtype=np.float32)), which Q32 "
                  "cannot imitate: the float32 rounding alone moves the output by up to 1.7e-6"},
}


def q32_round(value: Any) -> Any:
    """The nearest Q32 value of every float (Zinnia rounds v * 2^32 to the nearest integer)."""
    if isinstance(value, tuple):
        return tuple(q32_round(v) for v in value)
    array = np.asarray(value)
    if array.dtype.kind == "c":
        return q32_round(array.real) + 1j * q32_round(array.imag)
    if array.dtype.kind != "f":
        return value
    rounded = np.round(array * 2.0 ** 32) / 2.0 ** 32
    return rounded if array.ndim else float(rounded)


def _produced(function: Any, args: list[Any]) -> np.ndarray:
    copies = [copy.deepcopy(arg) for arg in args]
    with np.errstate(divide="raise", over="raise", invalid="raise", under="ignore"):
        output = function(*copies)
    return np.asarray(numbers(output) + numbers(copies))


def measured_difference(kernel: Path, function: str, how: str, inputs: list[list[Any]]) -> float:
    """The largest output difference the effect `how` causes, over the given inputs."""
    as_written = numpy_reference.load_kernel(kernel, function)
    if how == "q32_inputs":
        pairs = [(as_written, args, as_written, [q32_round(a) for a in args]) for args in inputs]
    elif how == "float64_buffers":
        widened = numpy_reference.load_kernel(
            kernel, function, lambda source: source.replace("dtype=np.float32", "dtype=np.float64"))
        pairs = [(as_written, args, widened, args) for args in inputs]
    else:
        raise ValueError(how)
    return max(float(np.max(np.abs(_produced(f, x) - _produced(g, y)))) for f, x, g, y in pairs)


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

