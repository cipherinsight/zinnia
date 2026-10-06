"""Source lowering: every rewrite of a kernel's Python source before the AST
transformer translates it for the Rust core.

The passes run in two stages, because the first needs the decorated Python
function object and the second only has source text.

**Decorator stage** (``@zk_circuit``, ``@zk_chip``, ``ZKChip.from_method``),
`read_decorated_function`:

1. `imports.check_module_level_imports`: a module-level import of a module
   other than ``numpy``/``math`` whose name the function uses → rejected.
2. `module_constants`: reads of module-level literal constants become the
   literals.
3. Circuits only, `nested_defs.lift_self_contained_defs`: a ``def`` in the
   circuit body that reads nothing of the circuit's own variables becomes a
   chip under its own name.

**Compile stage** (`lower_circuit_source`, from ``ZinniaCompiler``), on the
circuit only:

4. `imports.resolve_import_aliases`: imports in the source are checked by
   the same rule, and imported names are spelled ``np...``/``math...``.
5. `helpers`: the remaining local functions, lambdas and function aliases
   become generated chips that take their captured variables as arguments.
6. `callbacks`: ``map``, ``filter``, ``np.vectorize``, ``np.piecewise``,
   ``np.apply_along_axis`` and ``np.fromiter`` become calls of generated
   chips that loop over the elements.

Explicit chips and the chips the stages generate are not lowered further.

**During translation** the transformer (circuits and chips) applies
`iterators.desugar_for_iterator` to ``for`` loops over ``zip``,
``reversed`` and ``repeat``.

Every rewrite keeps Python's meaning or rejects the program with an error
naming the construct (`FunctionLoweringError`, `OutOfScopeImportError`).
"""
from __future__ import annotations

import inspect
from typing import Any, Dict, List, Tuple

from zinnia.compile.lowering.helpers import lower_circuit_source
from zinnia.compile.lowering.imports import (
    ALLOWED_IMPORT_ROOTS, OutOfScopeImportError, check_import_statement, check_module_level_imports,
)
from zinnia.compile.lowering.iterators import desugar_for_iterator
from zinnia.compile.lowering.module_constants import inline_module_constants, module_constants
from zinnia.compile.lowering.nested_defs import lift_self_contained_defs
from zinnia.compile.lowering.syntax import FunctionLoweringError, dedent_source

__all__ = [
    "ALLOWED_IMPORT_ROOTS",
    "FunctionLoweringError",
    "OutOfScopeImportError",
    "check_import_statement",
    "dedent_source",
    "desugar_for_iterator",
    "lower_circuit_source",
    "read_decorated_circuit",
    "read_decorated_function",
]


def read_decorated_function(method) -> str:
    """The source of a decorated function after the decorator stage (steps 1-2)."""
    raw_source = inspect.getsource(method)
    check_module_level_imports(method, raw_source)
    return inline_module_constants(raw_source, module_constants(method))


def read_decorated_circuit(method) -> Tuple[str, List[Tuple[str, str]]]:
    """The source of a ``@zk_circuit`` function after the decorator stage
    (steps 1-3), and the chips lifted from it as ``(name, source)`` pairs."""
    source = read_decorated_function(method)
    module_globals: Dict[str, Any] = getattr(method, "__globals__", {}) or {}
    return lift_self_contained_defs(source, module_globals)
