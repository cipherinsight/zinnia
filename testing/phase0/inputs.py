"""Random admitted inputs for the adequacy harness (`acceptance.md` §2.4).

An input is admitted when it satisfies the declared annotation and the
`requires` / `scalar_requires` facts. §2.4 asks for the *full* admitted
domain: full `int64` and `float64` element ranges, any length within a
declared bound, random masks. This module draws from it, and from a second,
deliberately benign domain.

# Why two modes

`coverage.md` §2 records that the front end aliases `Boolean` to `Integer`, so
a mask input is indistinguishable from an integer array in the annotation. A
full-range `int64` draw therefore turns `x[m]` into a fancy index with
astronomical indices, and essentially every draw fails. A failing draw still
tests G3 — some check must be violated — but it never exercises G1 and G2. So:

| Mode | Elements | What it exercises |
|---|---|---|
| `full` | the whole `int64` / `float64` range | G3, the failure side of adequacy |
| `benign` | small non-negative integers, small positive floats | G1 and G2, the success side |

Both are reported separately, as the brief for implementation step 3 asks.

# The one narrowing, and why

A bounded axis (`N<=1024`) is drawn within `[lo, min(hi, max_axis)]` rather
than up to `hi`: phase 0 unrolls, so a draw at the declared maximum costs the
harness the same quadratic blow-up that `coverage.md` §9 records for the
analysis itself. `max_axis` is a parameter of `draw`, reported with the
results.
"""

from __future__ import annotations

import contextlib
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from testing.phase0.terms import TermDag, UnboundVariable

#: Inclusive `int64` range, the element domain of §2.4.
INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1

#: The benign element domain: small enough to index, non-negative, non-zero
#: often enough that a division survives.
BENIGN_INT_MIN = 0
BENIGN_INT_MAX = 3


@dataclass(frozen=True)
class InputSpec:
    """One declared circuit input: its name, its annotation and its key."""

    name: str
    dt: Any
    key: str

    @staticmethod
    def of(ast: dict[str, Any]) -> list["InputSpec"]:
        return [
            InputSpec(name=declaration["name"],
                      dt=declaration["annotation"]["dt"],
                      key=f"|||{position}")
            for position, declaration in enumerate(ast.get("inputs", []))
        ]


@dataclass
class Domain:
    """The bounds phase 0 recorded, read back from its JSON (§2.4).

    `scalars` is keyed by the value key of the input, `symbols` by the shape
    symbol's name; both hold inclusive `(lo, hi)` pairs with `None` for an
    absent bound.
    """

    scalars: dict[str, tuple[int | None, int | None]] = field(default_factory=dict)
    symbols: dict[str, tuple[int | None, int | None]] = field(default_factory=dict)

    @staticmethod
    def of(analysis: dict[str, Any]) -> "Domain":
        """Read the ranges of `variables` whose provenance is an input.

        A bound is a *node* of the exported DAG, not a string: the
        `*_pretty` fields are the printer's output and are cut off at
        `PRETTY_BUDGET` (`implementation.md` §7), so reading a bound off
        them would tie the draws to the printer. A range bound is closed —
        it mentions no variable, by §3.4 — so the empty assignment
        evaluates it.
        """
        domain = Domain()
        dag = TermDag.from_json(analysis["terms"])
        bound_of = _closed_value_reader(dag)
        for variable in analysis.get("variables", []):
            bounds = (bound_of(variable.get("lo")), bound_of(variable.get("hi")))
            origin = variable["origin"]
            if origin["kind"] == "data_scalar":
                domain.scalars[origin["value"]["key"]] = bounds
            elif origin["kind"] == "shape_symbol":
                domain.symbols[origin["symbol"]] = bounds
        return domain


def _closed_value_reader(dag: TermDag) -> Any:
    """A reader for a closed integer node of `dag`, `None` when there is no
    node or the node turns out to mention a variable after all.
    """

    def read(node_id: Any) -> int | None:
        if node_id is None:
            return None
        try:
            return int(dag.evaluate(int(node_id), {}))
        except (UnboundVariable, ValueError, TypeError):
            return None

    return read


@dataclass
class DrawSettings:
    """What `draw` may do: the element domain and the axis cap."""

    mode: str = "full"  # "full" | "benign"
    max_axis: int = 16

    @property
    def benign(self) -> bool:
        return self.mode == "benign"


class Drawer:
    """Draws one admitted input tuple.

    A shape symbol is shared by name across inputs (`definition.md` §5.1), so
    its value is drawn once per tuple and reused.
    """

    def __init__(self, domain: Domain, rng: np.random.Generator,
                 settings: DrawSettings) -> None:
        self.domain = domain
        self.rng = rng
        self.settings = settings
        self.symbols: dict[str, int] = {}

    def tuple_for(self, specs: list[InputSpec]) -> list[Any]:
        """One value per declared input, symbols resolved consistently."""
        self.symbols = {}
        return [self.value(spec.dt, spec) for spec in specs]

    # ── values ────────────────────────────────────────────────────────

    def value(self, dt: Any, spec: InputSpec) -> Any:
        """The value of one annotation (`E-Input-*`, §5.1)."""
        if isinstance(dt, str):
            return self.scalar(dt, spec)
        kind, body = next(iter(dt.items()))
        if kind == "NDArray":
            return self.array(tuple(int(n) for n in body["shape"]), body["dtype"])
        if kind == "SymbolicNDArray":
            shape = tuple(self.axis(dim, body.get("bindings") or {})
                          for dim in body["shape"])
            return self.array(shape, body["dtype"])
        if kind == "DynamicNDArray":
            return self.array(self.dynamic_shape(body), body["dtype"])
        if kind in ("List", "Tuple"):
            elements = [self.value(element, spec) for element in body["elements"]]
            return elements if kind == "List" else tuple(elements)
        if kind == "PoseidonHashed":
            return self.value(body["dtype"], spec)
        raise NotImplementedError(f"input annotation `{kind}`")

    def scalar(self, dt: str, spec: InputSpec) -> Any:
        """A scalar input. A name that is also a shape symbol is that same
        quantity (`inputs.rs::scalar_input`), so it reuses the drawn value.
        """
        if dt == "Integer":
            if spec.name in self.domain.symbols:
                return self.symbol(spec.name)
            return self.integer(self.domain.scalars.get(spec.key, (None, None)))
        if dt == "Float":
            return self.real()
        if dt == "Boolean":
            return bool(self.rng.integers(0, 2))
        if dt == "Complex":
            return complex(self.real(), self.real())
        if dt == "String":
            return "".join(self.rng.choice(list("abcdefg"), size=4))
        if dt == "None":
            return None
        raise NotImplementedError(f"scalar input annotation `{dt}`")

    def integer(self, bounds: tuple[int | None, int | None]) -> int:
        """An integer inside the range phase 0 recorded, drawn from the mode's
        element domain and clamped into the range (§2.4).
        """
        lo, hi = bounds
        if self.settings.benign:
            low, high = BENIGN_INT_MIN, BENIGN_INT_MAX
        else:
            low, high = INT64_MIN, INT64_MAX
        low = max(low, lo) if lo is not None else low
        high = min(high, hi) if hi is not None else high
        if low > high:
            return low
        return int(self.rng.integers(low, high, endpoint=True, dtype=np.int64))

    def real(self) -> float:
        """A `float64`. The full mode spans the exponent range log-uniformly,
        which is the usable reading of "the full `float64` range": a uniform
        draw would be `1e308` with probability one.
        """
        if self.settings.benign:
            return float(self.rng.uniform(0.5, 4.0))
        sign = 1.0 if self.rng.integers(0, 2) else -1.0
        return sign * float(10.0 ** self.rng.uniform(-30.0, 30.0))

    def array(self, shape: tuple[int, ...], dtype: str) -> np.ndarray:
        """An array of the given shape, elements from the mode's domain."""
        size = int(np.prod(shape)) if shape else 1
        if dtype == "Float":
            flat = np.array([self.real() for _ in range(size)], dtype=np.float64)
        elif dtype == "Complex":
            flat = np.array([complex(self.real(), self.real()) for _ in range(size)],
                            dtype=np.complex128)
        else:
            flat = np.array([self.integer((None, None)) for _ in range(size)],
                            dtype=np.int64)
        return flat.reshape(shape)

    # ── shapes ────────────────────────────────────────────────────────

    def axis(self, dim: Any, bindings: dict[str, int]) -> int:
        """One axis of a `SymbolicNDArray` (`inputs.rs::symbolic_dim`)."""
        if isinstance(dim, int):
            return int(dim)
        if isinstance(dim, str):
            symbol, upper = dim, None
        else:
            symbol, upper = dim["symbol"], int(dim["upper"])
        if symbol in bindings:
            return int(bindings[symbol])
        return self.symbol(symbol, upper)

    def symbol(self, symbol: str, upper: int | None = None) -> int:
        """The shared value of a shape symbol, drawn once per input tuple."""
        if symbol in self.symbols:
            return self.symbols[symbol]
        lo, hi = self.domain.symbols.get(symbol, (0, None))
        low = max(0, lo if lo is not None else 0)
        candidates = [bound for bound in (hi, upper, self.settings.max_axis)
                      if bound is not None]
        high = min(candidates) if candidates else self.settings.max_axis
        value = low if low >= high else int(self.rng.integers(low, high,
                                                             endpoint=True))
        self.symbols[symbol] = value
        return value

    def dynamic_shape(self, body: dict[str, Any]) -> tuple[int, ...]:
        """`E-Input-VarLen`: one length per bounded axis, within the bound."""
        bounds = body.get("max_shape") or [body["max_length"]]
        return tuple(
            int(self.rng.integers(0, min(int(bound), self.settings.max_axis),
                                  endpoint=True))
            for bound in bounds
        )


def draw(
    analysis: dict[str, Any],
    ast: dict[str, Any],
    count: int,
    seed: int,
    settings: DrawSettings,
) -> list[list[Any]]:
    """`count` admitted input tuples for one circuit."""
    specs = InputSpec.of(ast)
    drawer = Drawer(Domain.of(analysis), np.random.default_rng(seed), settings)
    return [drawer.tuple_for(specs) for _ in range(count)]


# ---------------------------------------------------------------------------
# The hand-written DS-1000 inputs (`benchmarking/ds1000_case*/record.json`)
# ---------------------------------------------------------------------------


def record_inputs(
    kernel_path: Path, specs: list[InputSpec], limit: int = 2
) -> list[list[Any]]:
    """The `generate_test_case` fixtures of a DS-1000 case, as input tuples.

    `record.json` carries the original problem's `code_context`, whose
    `generate_test_case(i)` returns `(test_input, expected_result)`. Only the
    inputs are used here; the expected result is the baseline's business
    (`acceptance.md` §3), not adequacy's.

    A fixture whose shapes disagree with the kernel's annotation is dropped:
    the annotations were written for one of the record's test cases, so the
    other one is not an admitted input and adequacy says nothing about it.
    Returns an empty list when the case has no record, no builder, or no
    admitted fixture — these draws are extra, not required.
    """
    record = kernel_path.parent / "record.json"
    if not record.exists():
        return []
    builder = _exec_record(record).get("generate_test_case")
    if builder is None:
        return []
    tuples: list[list[Any]] = []
    for case in range(1, limit + 1):
        try:
            test_input, _ = builder(case)
        except Exception:  # noqa: BLE001 - a broken fixture is simply skipped
            continue
        values = [copy.deepcopy(value) for value in _as_tuple(test_input)]
        if admitted(specs, values):
            tuples.append(values)
    return tuples


def admitted(specs: list[InputSpec], values: list[Any]) -> bool:
    """Whether a hand-written input tuple satisfies the declared annotations."""
    if len(values) != len(specs):
        return False
    symbols: dict[str, int] = {}
    return all(_fits(spec.dt, value, symbols) for spec, value in zip(specs, values))


def _fits(dt: Any, value: Any, symbols: dict[str, int]) -> bool:
    """One value against one annotation: rank and extents, nothing finer."""
    if isinstance(dt, str):
        return not isinstance(value, np.ndarray)
    kind, body = next(iter(dt.items()))
    if kind == "NDArray":
        shape = tuple(int(n) for n in body["shape"])
        return isinstance(value, np.ndarray) and value.shape == shape
    if kind == "SymbolicNDArray":
        return _fits_symbolic(body, value, symbols)
    if kind == "DynamicNDArray":
        bounds = body.get("max_shape") or [body["max_length"]]
        return (isinstance(value, np.ndarray) and value.ndim == len(bounds)
                and all(n <= int(b) for n, b in zip(value.shape, bounds)))
    if kind in ("List", "Tuple"):
        elements = body["elements"]
        return (isinstance(value, (list, tuple)) and len(value) == len(elements)
                and all(_fits(e, v, symbols) for e, v in zip(elements, value)))
    if kind == "PoseidonHashed":
        return _fits(body["dtype"], value, symbols)
    return True


def _fits_symbolic(body: dict[str, Any], value: Any, symbols: dict[str, int]) -> bool:
    if not isinstance(value, np.ndarray) or value.ndim != len(body["shape"]):
        return False
    for axis, dim in enumerate(body["shape"]):
        extent = int(value.shape[axis])
        if isinstance(dim, int):
            if extent != dim:
                return False
            continue
        symbol = dim["symbol"] if isinstance(dim, dict) else dim
        upper = int(dim["upper"]) if isinstance(dim, dict) else None
        if upper is not None and extent > upper:
            return False
        if symbols.setdefault(symbol, extent) != extent:
            return False
    return True


def _exec_record(record: Path) -> dict[str, Any]:
    """Execute a record's `code_context` in a throwaway namespace.

    Every DS-1000 record imports `pandas` at the top even when the fixture is
    pure NumPy, and `pandas` is not a dependency of this repository. A stub
    module stands in for it; a fixture that really uses pandas then raises and
    is skipped, which is the same outcome as the import failing.
    """
    import json

    payload = json.loads(record.read_text(encoding="utf-8"))
    namespace: dict[str, Any] = {}
    with _stubbed("pandas"):
        try:
            exec(compile(payload.get("code_context", ""), str(record), "exec"),
                 namespace)
        except Exception:  # noqa: BLE001 - a fixture we cannot run is skipped
            return {}
    return namespace


@contextlib.contextmanager
def _stubbed(name: str) -> Any:
    """Bind `name` to an empty module for the duration, if it is missing."""
    import importlib.util
    import sys
    import types

    if name in sys.modules or importlib.util.find_spec(name) is not None:
        yield
        return
    sys.modules[name] = types.ModuleType(name)
    try:
        yield
    finally:
        sys.modules.pop(name, None)


def _as_tuple(test_input: Any) -> list[Any]:
    return list(test_input) if isinstance(test_input, tuple) else [test_input]
