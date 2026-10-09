"""The NumPy reference: a case's kernel body run as plain NumPy.

`kernel.py` starts with `from zinnia import *` and decorates its functions
with `@zk_circuit` / `@zk_chip`; its parameter annotations name Zinnia types.
We execute it with a stand-in `zinnia` module whose decorators are the
identity and whose type names are inert, so the body runs unchanged on NumPy
arrays built from the fixture.

The outputs are the entry function's return value, or, for a function that
returns `None` (NPBench/Pythran kernels that write into their arguments), the
array arguments the case set lists as written (`mutated_outputs`), in
parameter order, after the call: the outputs Zinnia, the zkVM ports and the
benchmark harness (`harness.outputs.call_outputs`) check. A complex value is
two floats, real part first, as Zinnia exposes it.
"""
from __future__ import annotations

import inspect
import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np


class _Inert:
    """Accepts any subscript or call, so annotations evaluate harmlessly."""

    def __getitem__(self, _item):
        return self

    def __call__(self, *args, **kwargs):
        return self


def _identity_decorator(fn=None, *args, **kwargs):
    if callable(fn):
        return fn
    return lambda f: f


def _condition_decorator(*_conditions):
    """`@requires(lambda ...: ...)` / `@ensures(...)`: the condition is a
    compile-time contract, never called; the decorated function is unchanged."""
    return lambda f: f


def _fake_zinnia() -> types.ModuleType:
    mod = types.ModuleType("zinnia")
    for name in ("zk_circuit", "zk_chip"):
        setattr(mod, name, _identity_decorator)
    for name in ("requires", "ensures"):
        setattr(mod, name, _condition_decorator)
    for name in ("Float", "Integer", "Boolean", "Complex", "Array", "Public", "Private"):
        setattr(mod, name, _Inert())
    mod.__all__ = [n for n in dir(mod) if not n.startswith("_")]
    return mod


def load_kernel(kernel_py: Path, function: str = "kernel"):
    """Returns the plain-Python entry function of a case."""
    saved = sys.modules.get("zinnia")
    sys.modules["zinnia"] = _fake_zinnia()
    try:
        namespace: dict[str, Any] = {"__name__": f"reference_{kernel_py.parent.name}"}
        exec(compile(kernel_py.read_text(), str(kernel_py), "exec"), namespace)
    finally:
        if saved is None:
            del sys.modules["zinnia"]
        else:
            sys.modules["zinnia"] = saved
    return namespace[function]


_DTYPES = {"float": np.float64, "int": np.int64, "bool": np.bool_, "complex": np.complex128}
_SCALARS = {"float": float, "int": int, "bool": bool, "complex": complex}


def _complex_values(value):
    """A fixture's complex value: each number is the pair `[real, imag]`."""
    if isinstance(value, list) and len(value) == 2 and all(isinstance(x, (int, float)) for x in value):
        return complex(value[0], value[1])
    return [_complex_values(v) for v in value]


def fixture_arguments(fixture: dict, input_name: str) -> list:
    """The kernel's positional arguments for one fixture input set."""
    values = fixture["inputs"][input_name]
    args = []
    for p in fixture["parameters"]:
        v = values[p["name"]]
        if p["dtype"] == "complex":
            v = _complex_values(v)
        if p["kind"] == "array":
            args.append(np.array(v, dtype=_DTYPES[p["dtype"]]).reshape(p["shape"]))
        else:
            args.append(_SCALARS[p["dtype"]](v))
    return args


def flatten_output(result) -> list[tuple[str, Any]]:
    """Flattens a return value into `(kind, value)` pairs, row-major,
    tuples and lists in order. `kind` is `float`, `int` or `bool`; a complex
    number is two floats, real part first."""
    out: list[tuple[str, Any]] = []

    def visit(x):
        if isinstance(x, (tuple, list)):
            for y in x:
                visit(y)
            return
        arr = np.asarray(x)
        if arr.dtype == np.bool_:
            kind = "bool"
        elif np.issubdtype(arr.dtype, np.integer):
            kind = "int"
        elif np.issubdtype(arr.dtype, np.floating):
            kind = "float"
        elif np.issubdtype(arr.dtype, np.complexfloating):
            for v in arr.ravel().tolist():
                out.extend([("float", v.real), ("float", v.imag)])
            return
        else:
            raise TypeError(f"unsupported output dtype {arr.dtype}")
        for v in arr.ravel().tolist():
            out.append((kind, v))

    visit(result)
    return out


def call_outputs(kernel, args: list, mutated_outputs: tuple[str, ...] | list[str] = ()):
    """The outputs of one call, unflattened: the return value, or the written
    arguments `mutated_outputs` (parameter names) for a function returning
    `None`. An array argument the call changes that the list leaves out is an
    error, so a missed output cannot pass."""
    before = [a.copy() if isinstance(a, np.ndarray) else a for a in args]
    result = kernel(*args)
    if result is not None:
        return result
    names = list(inspect.signature(kernel).parameters)
    for name, after, old in zip(names, args, before):
        if isinstance(after, np.ndarray) and name not in mutated_outputs and not np.array_equal(after, old):
            raise ValueError(f"the kernel writes argument {name!r}, which is not among its outputs")
    by_name = dict(zip(names, args))
    return tuple(by_name[name] for name in mutated_outputs)


def entry_outputs(entry, fixture: dict, input_name: str) -> list[tuple[str, Any]]:
    """The flattened NumPy outputs of an `entries.Entry` on one input set."""
    kernel = load_kernel(entry.kernel, entry.kernel_function)
    args = fixture_arguments(fixture, input_name)
    return flatten_output(call_outputs(kernel, args, entry.mutated_outputs))



def main() -> None:
    import argparse

    from .entries import resolve

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("case", help="case id or DS-1000 number (a DS-1000 case folder also works)")
    ap.add_argument("--fixture", type=Path)
    ap.add_argument("--input", default="B")
    a = ap.parse_args()
    entry = resolve(Path(a.case).name)
    fixture = json.loads((a.fixture or entry.fixture).read_text())
    print(json.dumps(entry_outputs(entry, fixture, a.input)))


if __name__ == "__main__":
    main()
