"""The ``@requires`` decorator.

``@requires`` is a *marker* decorator. It accepts a lambda expressing a
structural precondition. The lambda is never executed at runtime; the AST
transformer walks the decorated function's ``decorator_list`` at compile time
and extracts the lambda body to emit an ``IR::StructuralPredicate`` atom.

This file provides the runtime side: the decorator is a passthrough that
returns the function unchanged. The compile-time AST extraction lives in
``zinnia.compile.transformer._precondition``.
"""


def requires(precondition):
    """Decorate a circuit with a structural precondition.

    Usage::

        @zk_circuit
        @requires(lambda x, k: nnz(x) == k)
        def my_circuit(x, k):
            ...

    The argument must be a lambda expression whose body uses predicate
    markers from :mod:`zinnia.spec.predicates`. The lambda is never invoked
    at runtime; its AST is captured at compile time from the decorator's
    source.

    Returns the decorator that wraps the function (a passthrough — the
    semantic role lives in the AST transformer).
    """
    if not callable(precondition):
        raise TypeError(
            "@requires expects a single lambda argument; got "
            f"{type(precondition).__name__}"
        )

    def _decorate(fn):
        return fn

    return _decorate
