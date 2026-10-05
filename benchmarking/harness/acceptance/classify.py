"""Reason classes for programs the compiler rejects.

The classes are the ones the support account reports:

- `out_of_scope`: the program is outside the stated scope (imports other than
  numpy and math), decided before compiling by `kernels.out_of_scope_imports`,
  or reported so by the compiler;
- `unsupported_construct`: the compiler names a construct it does not implement;
- `no_provable_bound`: a size, index or precondition the compiler needs to know
  statically could not be proved;
- `solver_budget`: rejected for `no_provable_bound` at the default SMT budget,
  accepted with a larger one (decided by re-running, see `validate`);
- `compiler_error`: an internal error (panic, failed invariant, crash);
- `resource_limit`: the compile exceeded the time or memory limit;
- `unclassified`: none of the rules below matched; the raw message is kept so
  the rules can be refined. Never silently folded into another class.

The rules match the compiler's own message vocabulary (they are not keyed to
programs); the first matching rule wins.
"""

from __future__ import annotations

import re

RULES: tuple[tuple[str, re.Pattern], ...] = (
    ("out_of_scope", re.compile(r"Only supported NumPy imports are allowed", re.I)),
    ("no_provable_bound", re.compile(
        r"must be a compile-time constant|could not be proved in-bounds|requires precondition"
        r"|must be resolved to an exact value or finite bound|expected statically bounded"
        r"|is provably unsatisfiable", re.I)),
    # Internal errors before construct names: an internal error message can
    # mention an operation the compiler otherwise supports.
    ("compiler_error", re.compile(
        r"panicked|index out of bounds|capacity overflow|Variable `[^`]*` not found"
        r"|pop index out of range|compiler process|InternalZinniaException|IRBuilder::", re.I)),
    ("unsupported_construct", re.compile(
        r"not (yet )?(supported|implemented)|unsupported|Unknown function|does not (yet )?"
        r"support|only \w+ inputs are supported|supports [\w -]+ only|first slice supports"
        r"|currently require|cannot be used|is not allowed", re.I)),
)


def classify_compile_error(message: str) -> str:
    for reason_class, pattern in RULES:
        if pattern.search(message):
            return reason_class
    return "unclassified"
