"""Run a DS-1000 kernel body as plain NumPy, without Zinnia.

The kernel file is executed with `from zinnia import *` removed, the
decorators `@zk_circuit` / `@zk_chip` replaced by the identity, and the type
names (`Float`, `Integer`, `Array`, ...) replaced by inert stand-ins, so only
NumPy semantics remain. Arguments are copied first because kernels may update
their inputs in place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np


class _Annotation:
    """Accepts `Float`, `Float[Array, "8 6"]` and the like; means nothing."""

    def __getitem__(self, item: Any) -> "_Annotation":
        return self


def _identity(fn: Callable | None = None, **_kwargs: Any) -> Any:
    return fn if fn is not None else (lambda f: f)


def _requires(*_predicates: Any, **_kwargs: Any) -> Callable[[Callable], Callable]:
    """`@requires(lambda ...: ...)` states a precondition; NumPy ignores it."""
    return lambda fn: fn


def _namespace() -> dict[str, Any]:
    names = ["Float", "Integer", "Int", "Boolean", "Bool", "Complex", "Array", "DynArray", "Vec",
             "NDArray", "DynamicNDArray", "Public", "Private", "List", "Tuple"]
    namespace: dict[str, Any] = {name: _Annotation() for name in names}
    namespace.update({"zk_circuit": _identity, "zk_chip": _identity, "requires": _requires, "np": np})
    return namespace


def load_kernel(kernel_path: Path, function: str = "kernel") -> Callable[..., Any]:
    source = kernel_path.read_text(encoding="utf-8").replace("from zinnia import *", "")
    namespace = _namespace()
    exec(compile(source, str(kernel_path), "exec"), namespace)
    return namespace[function]


def run_kernel(kernel_path: Path, args: list[Any], function: str = "kernel") -> Any:
    copies = [np.array(a, copy=True) if isinstance(a, np.ndarray) else a for a in args]
    return run_kernel_in_place(kernel_path, copies, function)


def run_kernel_in_place(kernel_path: Path, args: list[Any], function: str = "kernel") -> Any:
    """Like run_kernel, but the kernel may update `args` (no copies)."""
    with np.errstate(divide="raise", over="raise", invalid="raise", under="ignore"):
        return load_kernel(kernel_path, function)(*args)
