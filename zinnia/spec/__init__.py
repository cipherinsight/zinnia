"""Zinnia precondition / specification surface.

Provides:
- ``requires(lambda x, k: nnz(x) == k)`` decorator for declaring structural
  preconditions on circuit inputs.
- Predicate marker symbols in :mod:`zinnia.spec.predicates`.

The decorator is a marker — its lambda is never executed. The AST transformer
extracts the lambda body from the function's ``decorator_list`` at compile
time and emits ``IR::StructuralPredicate`` atoms into the circuit IR.
"""

from zinnia.spec._decorator import requires
from zinnia.spec import predicates

__all__ = ["requires", "predicates"]
