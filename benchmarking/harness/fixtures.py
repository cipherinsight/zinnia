"""`fixture.json`: the inputs of one case (schema in `eval-redo/BRIEF.md`).

```json
{"case": "ds1000_case0300", "function": "kernel",
 "parameters": [{"name": "a", "kind": "array", "dtype": "float", "shape": [20]},
                {"name": "p", "kind": "scalar", "dtype": "int"}],
 "inputs": {"A": {"a": [...], "p": 25}, "B": {"a": [...], "p": 37}},
 "notes": "..."}
```

Input `A` is used for setup (key generation), input `B` for every timed proof
and the output check. Every input is private and every output public.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

SETUP_INPUT = "A"
PROVE_INPUT = "B"

_NUMPY_DTYPES = {"float": np.float64, "int": np.int64, "bool": np.bool_}
_SCALAR_TYPES = {"float": float, "int": int, "bool": bool}


class FixtureError(ValueError):
    """A fixture that does not satisfy the schema."""


@dataclass(frozen=True)
class Parameter:
    name: str
    kind: str  # "array" | "scalar"
    dtype: str  # "float" | "int" | "bool"
    shape: tuple[int, ...]


@dataclass(frozen=True)
class Fixture:
    case: str
    function: str
    parameters: tuple[Parameter, ...]
    inputs: Mapping[str, Mapping[str, Any]]
    path: Path
    sha256: str

    def values(self, input_name: str) -> dict[str, Any]:
        """The raw JSON values of one input, by parameter name."""
        try:
            return dict(self.inputs[input_name])
        except KeyError:
            raise FixtureError(f"{self.path}: no input {input_name!r}") from None

    def arguments(self, input_name: str) -> list[Any]:
        """The kernel's positional arguments for one input, in signature order.

        Arrays become NumPy arrays of the declared dtype and shape; scalars
        become Python `float`, `int` or `bool`.
        """
        values = self.values(input_name)
        return [_argument(param, values[param.name]) for param in self.parameters]


def _argument(param: Parameter, value: Any) -> Any:
    if param.kind == "scalar":
        return _SCALAR_TYPES[param.dtype](value)
    array = np.array(value, dtype=_NUMPY_DTYPES[param.dtype])
    if array.shape != param.shape:
        raise FixtureError(f"parameter {param.name!r}: value has shape {array.shape}, "
                           f"declared {param.shape}")
    return array


def _parameter(raw: Mapping[str, Any]) -> Parameter:
    kind = raw.get("kind")
    dtype = raw.get("dtype")
    if kind not in ("array", "scalar"):
        raise FixtureError(f"parameter {raw.get('name')!r}: kind must be array or scalar")
    if dtype not in _NUMPY_DTYPES:
        raise FixtureError(f"parameter {raw.get('name')!r}: dtype must be float, int or bool")
    shape = tuple(int(d) for d in raw.get("shape", ())) if kind == "array" else ()
    return Parameter(name=str(raw["name"]), kind=kind, dtype=dtype, shape=shape)


def load_fixture(path: Path) -> Fixture:
    """Load and validate one fixture file."""
    data = path.read_bytes()
    raw = json.loads(data)
    try:
        parameters = tuple(_parameter(p) for p in raw["parameters"])
        inputs = raw["inputs"]
        fixture = Fixture(case=str(raw["case"]), function=str(raw.get("function", "kernel")),
                          parameters=parameters, inputs=inputs, path=path,
                          sha256=hashlib.sha256(data).hexdigest())
    except KeyError as exc:
        raise FixtureError(f"{path}: missing field {exc}") from None
    _validate(fixture)
    return fixture


def _validate(fixture: Fixture) -> None:
    for name in (SETUP_INPUT, PROVE_INPUT):
        values = fixture.values(name)
        missing = [p.name for p in fixture.parameters if p.name not in values]
        if missing:
            raise FixtureError(f"{fixture.path}: input {name} lacks {missing}")
        fixture.arguments(name)  # checks shapes and dtypes
    if fixture.values(SETUP_INPUT) == fixture.values(PROVE_INPUT):
        raise FixtureError(f"{fixture.path}: inputs A and B must differ")
