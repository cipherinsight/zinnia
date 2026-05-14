"""AST-level extraction of ``@requires(...)`` and ``@zk_circuit(requires=...)``
preconditions from a circuit's ``decorator_list``.

Produces two spec shapes, each emitted into the ``ASTCircuit`` and
consumed downstream by the Rust IR generator:

  - ``ASTRequires { kind, args, op, bound }`` — structural-predicate
    calls (e.g., ``nnz(x) == k``, ``is_sorted(arr)``, ``forall(...)``).
  - ``ASTScalarRequires { term }`` — atomic scalar / arithmetic /
    logical preconditions whose body is built from comparisons,
    arithmetic, Boolean composition, and predicate calls. The ``term``
    field is a ContractTerm-shaped JSON dict that the Rust side
    deserializes into the existing ``formula::ContractTerm`` AST.
"""

import ast
from typing import List, Optional, Tuple

from zinnia.spec.predicates import PREDICATE_NAMES, _USER_PREDICATE_NAMES
from zinnia.debug.dbg_info import DebugInfo
from zinnia.debug.exception import InvalidProgramException


# Map ``ast.cmpop`` subclass → spec ``op`` string.
_COMPARE_OPS = {
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
}


def _render_rhs(node: ast.AST) -> str:
    """Render the right-hand side of a comparison as a flat string.

    Accepted shapes:
      - ``ast.Name`` → the name.
      - ``ast.Constant(int|float)`` → its literal repr.
      - ``ast.UnaryOp(USub, Constant)`` → "-N".
      - ``ast.BinOp`` with Name/Constant operands → "a + b" etc.

    Anything else raises ``ValueError``.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return repr(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return "-" + _render_rhs(node.operand)
    if isinstance(node, ast.BinOp):
        op_map = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
                  ast.FloorDiv: "//", ast.Mod: "%"}
        op = op_map.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported arithmetic in precondition rhs: "
                             f"{type(node.op).__name__}")
        return f"({_render_rhs(node.left)} {op} {_render_rhs(node.right)})"
    raise ValueError(f"unsupported rhs node `{type(node).__name__}` in "
                     f"precondition expression")


def _validate_predicate_args(args: List[ast.expr], dbg: DebugInfo) -> List[str]:
    """Render each predicate argument as a flat string.

    Each argument may be a parameter reference (``ast.Name``), an integer or
    float literal, a unary-minus on a literal, or a simple arithmetic
    expression — same grammar as the rhs of a comparison.

    The Rust side stores these as opaque strings; later cards interpret them
    against the predicate's expected signature (e.g., ``in_range(name, lo,
    hi)`` expects pos 0 = Name, pos 1/2 = literal).
    """
    rendered: List[str] = []
    for a in args:
        try:
            rendered.append(_render_rhs(a))
        except ValueError as e:
            raise InvalidProgramException(
                dbg,
                f"structural-predicate argument must be a parameter reference, "
                f"a numeric literal, or a simple arithmetic expression; got "
                f"`{ast.unparse(a)}` ({e})"
            )
    return rendered


def _build_spec(
    kind: str,
    args: List[ast.expr],
    op: Optional[str],
    rhs: Optional[ast.AST],
    dbg: DebugInfo,
) -> dict:
    """Build the ``ASTRequires`` dict for one precondition."""
    if kind not in PREDICATE_NAMES:
        raise InvalidProgramException(
            dbg,
            f"`{kind}` is not a registered structural predicate. Known "
            f"predicates: {sorted(PREDICATE_NAMES)}"
        )
    arg_names = _validate_predicate_args(args, dbg)
    bound: Optional[str] = None
    if rhs is not None:
        try:
            bound = _render_rhs(rhs)
        except ValueError as e:
            raise InvalidProgramException(dbg, str(e))
    return {
        "__class__": "ASTRequires",
        "kind": kind,
        "args": arg_names,
        "op": op,
        "bound": bound,
    }


# Inner element-predicates accepted inside `forall(arr, P)`. Each tuple
# is (arity, accepts_name_form). `accepts_name_form` is True when the
# user may write the inner predicate without parens (e.g., the nullary
# `forall(arr, is_nonneg)`).
_FORALL_INNER_PREDICATES = {
    "is_nonneg": (0, True),
    "lt_bound": (1, False),
    "eq_const": (1, False),
    "in_range": (2, False),
}


def _extract_forall(
    body: ast.Call,
    dbg: DebugInfo,
) -> List[dict]:
    """Special handling for `forall(arr, P)` preconditions.

    `P` is one of a closed set of element-predicate calls (or a bare
    `Name` for the nullary `is_nonneg`). The extractor flattens
    `forall(arr, lt_bound(B))` into a synthesized kind
    ``"forall_lt_bound"`` with combined args ``["arr", "B"]`` so the
    Rust side can register one entry per inner predicate without
    needing to special-case the nested-call IR shape.
    """
    if len(body.args) != 2:
        raise InvalidProgramException(
            dbg,
            f"forall(arr, P) takes exactly 2 arguments; got {len(body.args)}"
        )
    arr_node, inner_node = body.args[0], body.args[1]

    # Resolve the inner element-predicate name and any explicit args.
    inner_name: str
    inner_args: List[ast.expr]
    if isinstance(inner_node, ast.Call) and isinstance(inner_node.func, ast.Name):
        inner_name = inner_node.func.id
        inner_args = list(inner_node.args)
        if inner_node.keywords:
            raise InvalidProgramException(
                dbg,
                f"forall inner predicate `{inner_name}` does not accept "
                f"keyword arguments"
            )
    elif isinstance(inner_node, ast.Name):
        # Bare-name form: only allowed for nullary inner predicates.
        inner_name = inner_node.id
        inner_args = []
    else:
        raise InvalidProgramException(
            dbg,
            f"forall(arr, P): P must be an element-predicate call or "
            f"bare-name reference; got `{ast.unparse(inner_node)}`"
        )

    info = _FORALL_INNER_PREDICATES.get(inner_name)
    if info is None:
        raise InvalidProgramException(
            dbg,
            f"forall's inner predicate `{inner_name}` is not in the "
            f"closed element-predicate set: "
            f"{sorted(_FORALL_INNER_PREDICATES)}"
        )
    expected_arity, accepts_name_form = info
    if len(inner_args) != expected_arity:
        # Nullary may be written as bare-name (no Call) — accept either.
        if expected_arity == 0 and isinstance(inner_node, ast.Call) and len(inner_args) == 0:
            pass
        else:
            raise InvalidProgramException(
                dbg,
                f"forall's inner predicate `{inner_name}` expects "
                f"{expected_arity} argument(s); got {len(inner_args)}"
            )

    # Validate arr is a Name; flatten args = [arr_name, ...inner_args].
    if not isinstance(arr_node, ast.Name):
        raise InvalidProgramException(
            dbg,
            f"forall(arr, P): arr must be a parameter reference; got "
            f"`{ast.unparse(arr_node)}`"
        )

    kind = f"forall_{inner_name}"
    arg_nodes = [arr_node] + inner_args
    return [_build_spec(kind, arg_nodes, None, None, dbg)]


# ---------------------------------------------------------------------------
# Scalar precondition extraction
# ---------------------------------------------------------------------------
#
# When the lambda body is not a registered-predicate call (or a comparison
# whose left is one), we fall back to extracting a *scalar precondition* —
# an arbitrary Bool-typed expression over scalars, arithmetic, and
# predicate calls composed with `and` / `or` / `not`. The output is a
# ContractTerm-shaped JSON dict that the Rust side deserializes into the
# existing `optim::predicates::formula::ContractTerm`.

# AST cmp-op → ContractTerm CmpOp variant name (matches serde derive).
_CMP_OPS_CONTRACT = {
    ast.Eq: "Eq", ast.NotEq: "Ne",
    ast.Lt: "Lt", ast.LtE: "Le",
    ast.Gt: "Gt", ast.GtE: "Ge",
}

# AST binop → ContractTerm ArithOp variant name.
_ARITH_OPS_CONTRACT = {
    ast.Add: "Add",
    ast.Sub: "Sub",
    ast.Mult: "Mul",
    ast.Div: "Div",
    ast.FloorDiv: "FloorDiv",
    ast.Mod: "Mod",
    ast.Pow: "Pow",
}


def _build_var(name: str) -> dict:
    return {"Var": {"Input": name}}


def _build_lit_int(n: int) -> dict:
    return {"LitInt": n}


def _build_lit_bool(b: bool) -> dict:
    return {"LitBool": b}


def _build_arith(op: str, lhs: dict, rhs: dict) -> dict:
    return {"Arith": {"op": op, "lhs": lhs, "rhs": rhs}}


def _build_cmp(op: str, lhs: dict, rhs: dict) -> dict:
    return {"Cmp": {"op": op, "lhs": lhs, "rhs": rhs}}


def _build_bool_comb(op: str, operands: List[dict]) -> dict:
    return {"BoolComb": {"op": op, "operands": operands}}


def _build_not(inner: dict) -> dict:
    return {"Not": inner}


def _build_predicate_app(kind: str, args: List[dict]) -> dict:
    return {"PredicateApp": {"kind": kind, "args": args}}


def _build_int_term(
    node: ast.AST,
    fn_param_names: List[str],
    dbg: DebugInfo,
) -> dict:
    """Recursively build a ContractTerm for an Int-typed scalar expression."""
    if isinstance(node, ast.Name):
        if node.id not in fn_param_names:
            raise InvalidProgramException(
                dbg,
                f"scalar precondition references `{node.id}` which is not a "
                f"parameter of the decorated function"
            )
        return _build_var(node.id)
    if isinstance(node, ast.Constant):
        # `bool` is a subclass of `int`; check Bool before Int.
        if isinstance(node.value, bool):
            raise InvalidProgramException(
                dbg,
                f"scalar precondition Int-context expected an integer literal; "
                f"got Bool `{node.value}`"
            )
        if isinstance(node.value, int):
            return _build_lit_int(int(node.value))
        raise InvalidProgramException(
            dbg,
            f"scalar precondition Int-context expected an integer literal; "
            f"got `{node.value!r}` (a {type(node.value).__name__}). Float "
            f"literals are not yet supported in scalar preconditions."
        )
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            return _build_arith(
                "Sub", _build_lit_int(0),
                _build_int_term(node.operand, fn_param_names, dbg),
            )
        if isinstance(node.op, ast.UAdd):
            return _build_int_term(node.operand, fn_param_names, dbg)
        raise InvalidProgramException(
            dbg,
            f"unsupported unary operator `{type(node.op).__name__}` in "
            f"scalar precondition"
        )
    if isinstance(node, ast.BinOp):
        op = _ARITH_OPS_CONTRACT.get(type(node.op))
        if op is None:
            raise InvalidProgramException(
                dbg,
                f"unsupported arithmetic operator `{type(node.op).__name__}` "
                f"in scalar precondition; bitwise / shift operators require "
                f"Z3 BV theory and are deferred"
            )
        return _build_arith(
            op,
            _build_int_term(node.left, fn_param_names, dbg),
            _build_int_term(node.right, fn_param_names, dbg),
        )
    raise InvalidProgramException(
        dbg,
        f"scalar precondition Int-context cannot contain "
        f"`{ast.unparse(node)}` (a {type(node).__name__})"
    )


def _lower_membership(
    left: dict,
    container: ast.AST,
    negate: bool,
    fn_param_names: List[str],
    dbg: DebugInfo,
) -> dict:
    """Desugar ``x in container`` / ``x not in container`` into a
    ContractTerm. `left` is the already-built Int term for `x`.

    Supported containers:
      - ``range(hi)`` / ``range(lo, hi)`` / ``range(lo, hi, step)``
      - ``{c1, c2, …}`` (set literal)
      - ``[c1, c2, …]`` (list literal)
      - ``(c1, c2, …)`` (tuple literal)
    """
    if isinstance(container, ast.Call) \
            and isinstance(container.func, ast.Name) \
            and container.func.id == "range":
        args = container.args
        if len(args) == 1:
            lo_t = _build_lit_int(0)
            hi_t = _build_int_term(args[0], fn_param_names, dbg)
            step_t = None
        elif len(args) == 2:
            lo_t = _build_int_term(args[0], fn_param_names, dbg)
            hi_t = _build_int_term(args[1], fn_param_names, dbg)
            step_t = None
        elif len(args) == 3:
            lo_t = _build_int_term(args[0], fn_param_names, dbg)
            hi_t = _build_int_term(args[1], fn_param_names, dbg)
            step_t = _build_int_term(args[2], fn_param_names, dbg)
        else:
            raise InvalidProgramException(
                dbg, f"range() takes 1–3 arguments; got {len(args)}"
            )
        # `lo <= x < hi`
        clauses = [
            _build_cmp("Le", lo_t, left),
            _build_cmp("Lt", left, hi_t),
        ]
        if step_t is not None:
            # `(x - lo) % step == 0`
            shifted = _build_arith("Sub", left, lo_t)
            modded = _build_arith("Mod", shifted, step_t)
            clauses.append(_build_cmp("Eq", modded, _build_lit_int(0)))
        result = _build_bool_comb("And", clauses)
        return _build_not(result) if negate else result

    if isinstance(container, (ast.Set, ast.List, ast.Tuple)):
        elts = container.elts
        if not elts:
            # Empty set is empty membership; Python's `x in set()` is always False.
            return _build_lit_bool(True if negate else False)
        clauses = []
        for elt in elts:
            if not isinstance(elt, ast.Constant) or isinstance(elt.value, bool) \
                    or not isinstance(elt.value, int):
                raise InvalidProgramException(
                    dbg,
                    f"set / list / tuple membership requires literal integer "
                    f"members; got `{ast.unparse(elt)}`. Variable members "
                    f"are not yet supported."
                )
            clauses.append(_build_cmp("Eq", left, _build_lit_int(int(elt.value))))
        result = _build_bool_comb("Or", clauses) if len(clauses) > 1 else clauses[0]
        return _build_not(result) if negate else result

    raise InvalidProgramException(
        dbg,
        f"membership test `in {ast.unparse(container)}` requires a range() "
        f"call or a set / list / tuple literal"
    )


def _build_bool_term(
    node: ast.AST,
    fn_param_names: List[str],
    dbg: DebugInfo,
) -> dict:
    """Recursively build a ContractTerm for a Bool-typed expression."""
    # Boolean literal (must come before Int's Constant handler).
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return _build_lit_bool(bool(node.value))

    # Logical composition.
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            op = "And"
        elif isinstance(node.op, ast.Or):
            op = "Or"
        else:
            raise InvalidProgramException(
                dbg,
                f"unsupported BoolOp operator `{type(node.op).__name__}` in "
                f"scalar precondition"
            )
        operands = [_build_bool_term(v, fn_param_names, dbg) for v in node.values]
        return _build_bool_comb(op, operands)

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return _build_not(_build_bool_term(node.operand, fn_param_names, dbg))

    # Comparison, chained-comparison, or membership.
    if isinstance(node, ast.Compare):
        return _build_compare(node, fn_param_names, dbg)

    # Predicate call (bare unary).
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id == "forall":
            raise InvalidProgramException(
                dbg,
                "forall(arr, P) cannot appear inside a composite scalar "
                "precondition; use it as the top-level precondition or as "
                "a separate `@requires(...)`."
            )
        if node.func.id in _USER_PREDICATE_NAMES:
            args = [_build_int_term(a, fn_param_names, dbg) for a in node.args]
            return _build_predicate_app(node.func.id, args)
        raise InvalidProgramException(
            dbg,
            f"unknown function call `{node.func.id}` in scalar precondition; "
            f"only registered predicates are valid"
        )

    raise InvalidProgramException(
        dbg,
        f"scalar precondition Bool-context cannot contain "
        f"`{ast.unparse(node)}` (a {type(node).__name__})"
    )


def _build_compare(
    node: ast.Compare,
    fn_param_names: List[str],
    dbg: DebugInfo,
) -> dict:
    """Build a Bool ContractTerm from an `ast.Compare` node.

    Handles single comparisons, chained `lo <= x <= hi`, and membership.
    """
    if len(node.ops) == 1 and len(node.comparators) == 1:
        op = node.ops[0]
        rhs = node.comparators[0]
        # Membership (`x in container` / `x not in container`).
        if isinstance(op, (ast.In, ast.NotIn)):
            left = _build_int_term(node.left, fn_param_names, dbg)
            return _lower_membership(
                left, rhs, isinstance(op, ast.NotIn),
                fn_param_names, dbg,
            )
        # Standard scalar comparison.
        op_name = _CMP_OPS_CONTRACT.get(type(op))
        if op_name is None:
            raise InvalidProgramException(
                dbg,
                f"unsupported comparison operator `{type(op).__name__}`"
            )
        lhs = _build_int_term(node.left, fn_param_names, dbg)
        rhs_t = _build_int_term(rhs, fn_param_names, dbg)
        return _build_cmp(op_name, lhs, rhs_t)

    if len(node.ops) == 2 and len(node.comparators) == 2:
        # Chained `lo OP1 mid OP2 hi`.
        op1, op2 = node.ops
        if isinstance(op1, (ast.In, ast.NotIn)) or isinstance(op2, (ast.In, ast.NotIn)):
            raise InvalidProgramException(
                dbg,
                "membership (`in`) cannot appear in a chained comparison"
            )
        op1_name = _CMP_OPS_CONTRACT.get(type(op1))
        op2_name = _CMP_OPS_CONTRACT.get(type(op2))
        if op1_name is None or op2_name is None:
            raise InvalidProgramException(
                dbg,
                f"unsupported chained comparison `{type(op1).__name__}` / "
                f"`{type(op2).__name__}`"
            )
        left_t = _build_int_term(node.left, fn_param_names, dbg)
        mid_t = _build_int_term(node.comparators[0], fn_param_names, dbg)
        right_t = _build_int_term(node.comparators[1], fn_param_names, dbg)
        return _build_bool_comb("And", [
            _build_cmp(op1_name, left_t, mid_t),
            _build_cmp(op2_name, mid_t, right_t),
        ])

    raise InvalidProgramException(
        dbg,
        f"scalar precondition supports at most two chained comparison ops; "
        f"got {len(node.ops)} in `{ast.unparse(node)}`"
    )


def _is_registered_predicate_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _USER_PREDICATE_NAMES
    )


def _structural_compare_left_is_predicate(node: ast.Compare) -> bool:
    """Heuristic: a `Compare` whose left side is a registered predicate
    call and whose right side is a `Name`/`Constant`/simple arith fits
    the existing structural-predicate path. Everything else falls
    through to scalar extraction.

    This preserves the existing `nnz(x) == k`-style preconditions
    without rewriting them through the scalar path.
    """
    return (
        len(node.ops) == 1
        and len(node.comparators) == 1
        and _is_registered_predicate_call(node.left)
        # Predicate name is in the user-surface set; not a synthesized
        # `forall_*` IR kind.
        and node.left.func.id != "forall"
    )


def _extract_from_lambda_body(
    body: ast.AST,
    dbg: DebugInfo,
) -> List[dict]:
    """Extract precondition specs from a lambda body.

    Recognized shapes (in order of precedence):
      - ``forall(arr, P)`` — synthesized `forall_<inner>` structural kind.
      - ``predicate(args...)`` — unary structural predicate call.
      - ``predicate(args...) <op> rhs`` — structural predicate comparison.
      - Any other Bool-typed expression — falls through to the scalar
        precondition path (see `_build_bool_term`).
    """
    # Special-case: forall(arr, P).
    if isinstance(body, ast.Call) \
            and isinstance(body.func, ast.Name) \
            and body.func.id == "forall":
        if body.keywords:
            raise InvalidProgramException(
                dbg,
                "forall(arr, P) does not accept keyword arguments"
            )
        return _extract_forall(body, dbg)

    # Bare structural-predicate call.
    if _is_registered_predicate_call(body) and body.func.id != "forall":
        if body.keywords:
            raise InvalidProgramException(
                dbg,
                f"predicate `{body.func.id}` does not accept keyword arguments"
            )
        return [_build_spec(body.func.id, body.args, None, None, dbg)]

    # Structural-predicate comparison (`nnz(x) == k`, etc.).
    if isinstance(body, ast.Compare) and _structural_compare_left_is_predicate(body):
        op_cls = type(body.ops[0])
        op_str = _COMPARE_OPS.get(op_cls)
        if op_str is None:
            raise InvalidProgramException(
                dbg,
                f"unsupported comparison operator `{op_cls.__name__}` in "
                f"structural precondition"
            )
        left = body.left
        if left.keywords:
            raise InvalidProgramException(
                dbg,
                f"predicate `{left.func.id}` does not accept keyword arguments"
            )
        return [_build_spec(
            left.func.id, left.args, op_str, body.comparators[0], dbg
        )]

    # Everything else: scalar precondition. Build a ContractTerm.
    term = _build_bool_term(body, [], dbg) if False else None
    # We need fn_param_names here; the caller passes them via a closure-
    # style — but the existing `_extract_from_lambda_body` doesn't take
    # them. Plumb through _extract_one_lambda instead (see below).
    raise InvalidProgramException(
        dbg,
        "internal: scalar precondition extraction must be invoked via "
        "_extract_from_lambda_body_with_params"
    )


def _extract_from_lambda_body_with_params(
    body: ast.AST,
    fn_param_names: List[str],
    dbg: DebugInfo,
) -> List[dict]:
    """Same as `_extract_from_lambda_body` but with the surrounding
    function's parameter names available for scalar-extraction Name
    validation. Always preferred over the no-params variant.
    """
    # Structural-predicate fast paths first.
    if isinstance(body, ast.Call) \
            and isinstance(body.func, ast.Name) \
            and body.func.id == "forall":
        if body.keywords:
            raise InvalidProgramException(
                dbg,
                "forall(arr, P) does not accept keyword arguments"
            )
        return _extract_forall(body, dbg)

    if _is_registered_predicate_call(body) and body.func.id != "forall":
        if body.keywords:
            raise InvalidProgramException(
                dbg,
                f"predicate `{body.func.id}` does not accept keyword arguments"
            )
        return [_build_spec(body.func.id, body.args, None, None, dbg)]

    if isinstance(body, ast.Compare) and _structural_compare_left_is_predicate(body):
        op_cls = type(body.ops[0])
        op_str = _COMPARE_OPS.get(op_cls)
        if op_str is None:
            raise InvalidProgramException(
                dbg,
                f"unsupported comparison operator `{op_cls.__name__}` in "
                f"structural precondition"
            )
        left = body.left
        if left.keywords:
            raise InvalidProgramException(
                dbg,
                f"predicate `{left.func.id}` does not accept keyword arguments"
            )
        return [_build_spec(
            left.func.id, left.args, op_str, body.comparators[0], dbg
        )]

    # Scalar precondition: build a ContractTerm.
    term = _build_bool_term(body, fn_param_names, dbg)
    return [{
        "__class__": "ASTScalarRequires",
        "term": term,
    }]


def _extract_one_lambda(
    lam: ast.Lambda,
    fn_param_names: List[str],
    dbg: DebugInfo,
) -> List[dict]:
    """Extract precondition specs from one lambda.

    Validates that the lambda's parameter names are a subset of the
    decorated function's parameter names.
    """
    lam_param_names = [a.arg for a in lam.args.args]
    for p in lam_param_names:
        if p not in fn_param_names:
            raise InvalidProgramException(
                dbg,
                f"precondition lambda parameter `{p}` is not a parameter of "
                f"the decorated function (parameters: {fn_param_names})"
            )
    return _extract_from_lambda_body_with_params(lam.body, fn_param_names, dbg)


def extract_preconditions(
    func_node: ast.FunctionDef,
    source_code: str,
    method_name: str,
) -> List[dict]:
    """Walk a function's ``decorator_list`` and extract precondition specs.

    Recognized decorator shapes:
      - ``@requires(lambda ...)``
      - ``@zk_circuit(requires=[lambda ..., lambda ...])`` (or with other
        kwargs alongside)

    Returns the flat list of ``ASTRequires`` dicts in source order.
    """
    fn_param_names = [a.arg for a in func_node.args.args]
    specs: List[dict] = []

    for deco in func_node.decorator_list:
        dbg = _dbg_for(deco, source_code, method_name)

        # @requires(lambda ...)
        if isinstance(deco, ast.Call) and \
                isinstance(deco.func, ast.Name) and deco.func.id == "requires":
            if len(deco.args) != 1 or deco.keywords:
                raise InvalidProgramException(
                    dbg,
                    "@requires(...) takes exactly one positional argument (a "
                    "lambda); no keyword arguments allowed."
                )
            lam = deco.args[0]
            if not isinstance(lam, ast.Lambda):
                raise InvalidProgramException(
                    dbg,
                    f"@requires(...) argument must be a lambda; got "
                    f"`{type(lam).__name__}`"
                )
            specs.extend(_extract_one_lambda(lam, fn_param_names, dbg))

        # @zk_circuit(requires=[lambda ..., ...])
        elif isinstance(deco, ast.Call) and \
                isinstance(deco.func, ast.Name) and deco.func.id == "zk_circuit":
            for kw in deco.keywords:
                if kw.arg != "requires":
                    continue
                if not isinstance(kw.value, ast.List):
                    raise InvalidProgramException(
                        dbg,
                        f"@zk_circuit(requires=...) must be a list of "
                        f"lambdas; got `{type(kw.value).__name__}`"
                    )
                for lam in kw.value.elts:
                    if not isinstance(lam, ast.Lambda):
                        raise InvalidProgramException(
                            _dbg_for(lam, source_code, method_name),
                            f"@zk_circuit(requires=[...]) entries must be "
                            f"lambdas; got `{type(lam).__name__}`"
                        )
                    specs.extend(_extract_one_lambda(
                        lam, fn_param_names,
                        _dbg_for(lam, source_code, method_name)
                    ))

    return specs


def _dbg_for(node: ast.AST, source_code: str, method_name: str) -> DebugInfo:
    return DebugInfo(
        method_name, source_code, True,
        getattr(node, "lineno", 0),
        getattr(node, "col_offset", 0),
        getattr(node, "end_lineno", 0),
        getattr(node, "end_col_offset", 0),
    )
