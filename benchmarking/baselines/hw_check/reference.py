"""The NumPy reference: a case's kernel body run as plain NumPy.

`kernel.py` starts with `from zinnia import *` and decorates its functions
with `@zk_circuit` / `@zk_chip`; its parameter annotations name Zinnia types.
We execute it with a stand-in `zinnia` module whose decorators are the
identity and whose type names are inert, so the body runs unchanged on NumPy
arrays built from the fixture.
"""
from __future__ import annotations

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


def _fake_zinnia() -> types.ModuleType:
    mod = types.ModuleType("zinnia")
    for name in ("zk_circuit", "zk_chip", "requires", "ensures"):
        setattr(mod, name, _identity_decorator)
    for name in ("Float", "Integer", "Boolean", "Array", "Public", "Private"):
        setattr(mod, name, _Inert())
    mod.__all__ = [n for n in dir(mod) if not n.startswith("_")]
    return mod


def load_kernel(kernel_py: Path):
    """Returns the plain-Python `kernel` function of a case."""
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
    return namespace["kernel"]


_DTYPES = {"float": np.float64, "int": np.int64, "bool": np.bool_}


def fixture_arguments(fixture: dict, input_name: str) -> list:
    """The kernel's positional arguments for one fixture input set."""
    values = fixture["inputs"][input_name]
    args = []
    for p in fixture["parameters"]:
        v = values[p["name"]]
        if p["kind"] == "array":
            args.append(np.array(v, dtype=_DTYPES[p["dtype"]]).reshape(p["shape"]))
        else:
            args.append({"float": float, "int": int, "bool": bool}[p["dtype"]](v))
    return args


def flatten_output(result) -> list[tuple[str, Any]]:
    """Flattens a return value into `(kind, value)` pairs, row-major,
    tuples and lists in order. `kind` is `float`, `int` or `bool`."""
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
        else:
            raise TypeError(f"unsupported output dtype {arr.dtype}")
        for v in arr.ravel().tolist():
            out.append((kind, v))

    visit(result)
    return out


def reference_outputs(case_dir: Path, fixture: dict, input_name: str) -> list[tuple[str, Any]]:
    kernel = load_kernel(case_dir / "kernel.py")
    args = fixture_arguments(fixture, input_name)
    return flatten_output(kernel(*args))


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("case_dir", type=Path)
    ap.add_argument("--fixture", type=Path)
    ap.add_argument("--input", default="B")
    a = ap.parse_args()
    fixture = json.loads((a.fixture or a.case_dir / "fixture.json").read_text())
    print(json.dumps(reference_outputs(a.case_dir, fixture, a.input)))


if __name__ == "__main__":
    main()
