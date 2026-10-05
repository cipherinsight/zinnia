"""Loading, classifying and compiling `kernel.py` files, and the scope rule."""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

#: The scope rule (author's decision, applied to every dataset): a kernel that
#: imports any module other than `numpy` (and its submodules) and `math` is out
#: of scope. `zinnia` is the language's own import and is always allowed.
ALLOWED_TOP_LEVEL_MODULES = frozenset({"numpy", "math", "zinnia"})


def imported_modules(path: Path) -> list[str]:
    """Top-level module names of every import statement in the file."""
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import: a sibling module, never in scope
                names.append("." * node.level + (node.module or ""))
            elif node.module:
                names.append(node.module.split(".")[0])
    return names


def out_of_scope_imports(path: Path) -> list[str]:
    """The imports that put this kernel out of scope (empty = in scope)."""
    return sorted({n for n in imported_modules(path) if n not in ALLOWED_TOP_LEVEL_MODULES})


def function_roles(path: Path) -> dict[str, str]:
    """`circuit`, `chip` or `plain` for every top-level function of the file."""
    tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    roles: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        decorators = {ast.unparse(dec) for dec in node.decorator_list}
        if any(d == "zk_circuit" or d.startswith("zk_circuit(") for d in decorators):
            roles[node.name] = "circuit"
        elif any(d == "zk_chip" or d.startswith("zk_chip(") for d in decorators):
            roles[node.name] = "chip"
        else:
            roles[node.name] = "plain"
    return roles


def load_kernel_module(path: Path) -> ModuleType:
    """Import a kernel file under a module name derived from its folder.

    The module is registered in `sys.modules` so that chips and helpers
    defined in the same file resolve during compilation.
    """
    module_name = "zinnia_benchmark_" + path.parent.name
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def compile_kernel(path: Path, function: str) -> Any:
    """Compile one `@zk_circuit` function; returns a `ZKCompiledProgram`."""
    from zinnia import ZKCircuit

    module = load_kernel_module(path)
    return ZKCircuit.from_method(getattr(module, function)).compile()
