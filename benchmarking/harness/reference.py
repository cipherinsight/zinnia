"""The NumPy reference: a kernel's body run as plain NumPy.

The `@zk_circuit` decorator (and `@zk_chip`, `@zk_external`) becomes the
identity, Zinnia's annotation names stay importable, and the function is called
on copies of the inputs (kernels may mutate their arguments). No per-case
overrides: a kernel whose body is not valid NumPy has no reference, and the
harness reports that.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _identity_decorator(method=None, **_kwargs):
    if method is None:
        return lambda m: m
    return method


def kernel_namespace() -> dict[str, Any]:
    import zinnia

    namespace = {name: getattr(zinnia, name) for name in dir(zinnia) if not name.startswith("_")}
    namespace["np"] = np
    namespace["zk_circuit"] = _identity_decorator
    namespace["zk_chip"] = _identity_decorator
    namespace["zk_external"] = _identity_decorator
    return namespace


def run_kernel_as_numpy(kernel_path: Path, function: str, args: list[Any]) -> Any:
    """Execute the kernel file with the shims above and call `function(*args)`."""
    source = kernel_path.read_text()
    # The real import would rebind the shims; Zinnia's names are already there.
    source = source.replace("from zinnia import *", "")
    namespace = kernel_namespace()
    exec(compile(source, str(kernel_path), "exec"), namespace)  # noqa: S102 - our own datasets
    call_args = [np.array(a, copy=True) if isinstance(a, np.ndarray) else a for a in args]
    return namespace[function](*call_args)
