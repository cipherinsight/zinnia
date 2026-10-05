import ast

from zinnia.debug.exception import InvalidCircuitStatementException, \
    InvalidProgramException, InvalidAssignStatementException, InvalidAnnotationException, UnsupportedOperatorException, \
    UnsupportedConstantLiteralException, \
    UnsupportedLangFeatureException
from zinnia.debug.dbg_info import DebugInfo


# Type name aliases → canonical ZinniaType variant name.
_TYPE_ALIASES = {
    "Integer": "Integer", "int": "Integer", "Int": "Integer", "integer": "Integer",
    "Boolean": "Integer", "bool": "Integer", "Bool": "Integer", "boolean": "Integer",
    "Float": "Float", "float": "Float",
    "Complex": "Complex", "complex": "Complex",
    "String": "String", "str": "String",
    "None": "None",
    "Class": "Class",
    "NDArray": "NDArray",
    "SymbolicNDArray": "SymbolicNDArray",
    "DynamicNDArray": "DynamicNDArray",
    "List": "List", "list": "List",
    "Tuple": "Tuple", "tuple": "Tuple",
    "PoseidonHashed": "PoseidonHashed",
}

# Scalar types that need no arguments. Note: every Boolean alias collapses to
# "Integer" via `_TYPE_ALIASES`, so "Boolean" never reaches `make_type_dict` as
# a resolved variant and is intentionally absent here.
_SCALAR_TYPES = {"Integer", "Float", "Complex", "String", "None", "Class"}
_DTYPE_FRONT_ALIASES = {
    "Float": "Float",
    "Int": "Integer",
    "Integer": "Integer",
    "Bool": "Boolean",
    "Boolean": "Boolean",
    "Complex": "Complex",
}


def _parse_shape_expr_token(token: str):
    tree = ast.parse(token, mode="eval")

    def conv(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
            return ("const", node.value)
        if isinstance(node, ast.Name) and node.id.isidentifier():
            return ("symbol", node.id)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return ("neg", conv(node.operand))
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv)):
            op = {
                ast.Add: "+",
                ast.Sub: "-",
                ast.Mult: "*",
                ast.FloorDiv: "//",
            }[type(node.op)]
            return (op, conv(node.left), conv(node.right))
        raise ValueError(f"unsupported shape expression `{token}`")

    return conv(tree.body)


def _eval_shape_expr(expr, bindings: dict[str, int]):
    kind = expr[0]
    if kind == "const":
        return expr[1]
    if kind == "symbol":
        return bindings.get(expr[1])
    if kind == "neg":
        val = _eval_shape_expr(expr[1], bindings)
        return None if val is None else -val
    lhs = _eval_shape_expr(expr[1], bindings)
    rhs = _eval_shape_expr(expr[2], bindings)
    if lhs is None or rhs is None:
        return None
    if kind == "+":
        return lhs + rhs
    if kind == "-":
        return lhs - rhs
    if kind == "*":
        return lhs * rhs
    if kind == "//":
        if rhs == 0:
            raise ValueError("shape expression divides by zero")
        return lhs // rhs
    raise ValueError(f"unknown shape expression operator `{kind}`")


def _shape_expr_symbols(expr) -> set[str]:
    kind = expr[0]
    if kind == "symbol":
        return {expr[1]}
    if kind in {"const"}:
        return set()
    if kind == "neg":
        return _shape_expr_symbols(expr[1])
    return _shape_expr_symbols(expr[1]) | _shape_expr_symbols(expr[2])


def _shape_expr_to_string(expr) -> str:
    kind = expr[0]
    if kind == "const":
        return str(expr[1])
    if kind == "symbol":
        return expr[1]
    if kind == "neg":
        return f"-{_shape_expr_to_string(expr[1])}"
    return f"{_shape_expr_to_string(expr[1])}{kind}{_shape_expr_to_string(expr[2])}"


def make_type_dict(dbg, typename, args=None):
    """Create a type descriptor dict matching Rust's ZinniaType serde format.

    Scalars:  "Integer", "Float", etc.
    Compound: {"NDArray": {"shape": [2,3], "dtype": "Integer"}}, etc.
    """
    if args is None:
        args = ()

    variant = _TYPE_ALIASES.get(typename)
    if variant is None:
        raise InvalidAnnotationException(dbg, f'`{typename}` is not a valid type name')

    if variant in _SCALAR_TYPES:
        return variant  # Just the string

    if variant == "NDArray":
        if len(args) < 2:
            raise InvalidAnnotationException(dbg, "NDArray requires at least 2 arguments (dtype and shape dimensions).")
        dtype = args[0]
        shape = []
        for a in args[1:]:
            if not isinstance(a, int):
                raise InvalidAnnotationException(dbg, "NDArray shape dimensions must be integers.")
            shape.append(a)
        return {"NDArray": {"shape": shape, "dtype": dtype}}

    if variant == "SymbolicNDArray":
        if len(args) < 2:
            raise InvalidAnnotationException(dbg, "SymbolicNDArray requires at least 2 arguments (dtype and shape dimensions).")
        dtype = args[0]
        shape = []
        bindings = {}
        for a in args[1:]:
            if isinstance(a, int):
                shape.append(a)
            elif isinstance(a, str) and a.isidentifier():
                shape.append(a)
            elif isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], str) and isinstance(a[1], int):
                shape.append(a[0])
                bindings[a[0]] = a[1]
            elif (
                isinstance(a, tuple)
                and len(a) == 3
                and a[0] == "bound"
                and isinstance(a[1], str)
                and isinstance(a[2], int)
            ):
                shape.append({"symbol": a[1], "upper": a[2]})
            else:
                raise InvalidAnnotationException(
                    dbg,
                    "SymbolicNDArray shape dimensions must be integers, unresolved symbols, exact bindings, or bounded symbols."
                )
        return {"SymbolicNDArray": {"shape": shape, "dtype": dtype, "bindings": bindings}}

    if variant == "DynamicNDArray":
        if len(args) != 3:
            raise InvalidAnnotationException(dbg, "DynamicNDArray requires exactly 3 arguments (dtype, max_length, max_rank).")
        dtype = args[0]
        if not isinstance(args[1], int) or not isinstance(args[2], int):
            raise InvalidAnnotationException(dbg, "DynamicNDArray max_length and max_rank must be integers.")
        return {"DynamicNDArray": {"dtype": dtype, "max_length": args[1], "max_rank": args[2]}}

    if variant == "List":
        if len(args) < 1:
            raise InvalidAnnotationException(dbg, "List requires at least 1 argument.")
        return {"List": {"elements": list(args)}}

    if variant == "Tuple":
        if len(args) < 1:
            raise InvalidAnnotationException(dbg, "Tuple requires at least 1 argument.")
        return {"Tuple": {"elements": list(args)}}

    if variant == "PoseidonHashed":
        if len(args) != 1:
            raise InvalidAnnotationException(dbg, "PoseidonHashed requires exactly 1 argument.")
        return {"PoseidonHashed": {"dtype": args[0]}}

    raise InvalidAnnotationException(dbg, f'`{typename}` is not a valid type name')


class ZinniaBaseASTTransformer(ast.NodeTransformer):
    def __init__(self, source_code: str, method_name: str):
        super().__init__()
        self.source_code = source_code
        self.method_name = method_name
        self.shape_symbol_bindings = {}
        self.shape_symbol_names = set()
        self.shape_symbol_bounds = {}

    def get_op_name_from_node(self, node) -> str:
        if isinstance(node, ast.Add):
            return "add"
        elif isinstance(node, ast.Sub):
            return "sub"
        elif isinstance(node, ast.Div):
            return "div"
        elif isinstance(node, ast.Mult):
            return "mul"
        elif isinstance(node, ast.MatMult):
            return "mat_mul"
        elif isinstance(node, ast.FloorDiv):
            return "floor_div"
        elif isinstance(node, ast.Mod):
            return "mod"
        if isinstance(node, ast.Not):
            return "not"
        elif isinstance(node, ast.USub):
            return "usub"
        elif isinstance(node, ast.UAdd):
            return "uadd"
        elif isinstance(node, ast.Pow):
            return "pow"
        elif isinstance(node, ast.BitAnd):
            return "bit_and"
        elif isinstance(node, ast.BitOr):
            return "bit_or"
        elif isinstance(node, ast.BitXor):
            return "bit_xor"
        elif isinstance(node, ast.LShift):
            return "shl"
        elif isinstance(node, ast.RShift):
            return "shr"
        elif isinstance(node, ast.Invert):
            return "invert"
        raise UnsupportedOperatorException(self.get_dbg(node), f"Invalid operator {type(node).__name__} in circuit.")

    def get_comp_op_name_from_node(self, node) -> str:
        if isinstance(node, ast.GtE):
            return "gte"
        elif isinstance(node, ast.LtE):
            return "lte"
        elif isinstance(node, ast.Gt):
            return "gt"
        elif isinstance(node, ast.Lt):
            return "lt"
        elif isinstance(node, ast.Eq):
            return "eq"
        elif isinstance(node, ast.NotEq):
            return "ne"
        elif isinstance(node, ast.In):
            return "in"
        elif isinstance(node, ast.NotIn):
            return "not_in"
        raise UnsupportedOperatorException(self.get_dbg(node),
                                           f"Invalid compare operator {type(node).__name__} in circuit.")

    def get_dbg(self, node) -> DebugInfo:
        raise NotImplementedError(
            "get_dbg is abstract; concrete transformer subclasses must override it."
        )

    def visit(self, node):
        if isinstance(node, ast.FunctionDef):
            return self.visit_FunctionDef(node)
        raise InvalidProgramException(None, "Invalid code passed to the compiler! The program must be a function.")

    def visit_For(self, node: ast.For):
        # Desugar iterator-shaped builtins that only make sense in for-loop
        # position: zip, reversed, itertools.repeat. Rewrite to an
        # index-based for over a range and synthesize per-iteration
        # bindings, then re-enter visit_For on the rewritten node.
        rewritten = self._desugar_iterator_builtin(node)
        if rewritten is not None:
            return self.visit_For(rewritten)

        target = self.visit_assign_target(node.target)
        iter_expr = self.visit_expr(node.iter)
        return {"__class__": "ASTForInStatement",
                "target": target, "iter_expr": iter_expr,
                "block": self.visit_block(node.body), "orelse": self.visit_block(node.orelse)}

    def _desugar_iterator_builtin(self, node: ast.For):
        """If ``node.iter`` is a zip / reversed / itertools.repeat call,
        return a rewritten ``ast.For`` with an equivalent range-based
        iteration. Returns ``None`` otherwise.
        """
        it = node.iter
        if not isinstance(it, ast.Call):
            return None

        # itertools.repeat(value, n) — supported in iter position only.
        # Match `it.repeat(...)`, `itertools.repeat(...)`, or bare `repeat(...)`.
        is_repeat = (
            (isinstance(it.func, ast.Attribute) and it.func.attr == "repeat")
            or (isinstance(it.func, ast.Name) and it.func.id == "repeat")
        )
        if is_repeat and len(it.args) == 2:
            value_expr, count_expr = it.args
            new_for = ast.For(
                target=ast.Name(id="_", ctx=ast.Store()),
                iter=ast.Call(
                    func=ast.Name(id="range", ctx=ast.Load()),
                    args=[count_expr], keywords=[],
                ),
                body=[
                    ast.Assign(
                        targets=[node.target],
                        value=value_expr,
                    ),
                    *node.body,
                ] if not (isinstance(node.target, ast.Name) and node.target.id == "_")
                  else list(node.body),
                orelse=node.orelse,
            )
            ast.copy_location(new_for, node)
            ast.fix_missing_locations(new_for)
            return new_for

        # reversed(iterable) in iter position.
        if isinstance(it.func, ast.Name) and it.func.id == "reversed" and len(it.args) == 1:
            inner = it.args[0]
            # Generate a fresh index name. Use a simple counter on the
            # transformer instance to avoid collisions across nested loops.
            idx_name = self._fresh_name("__rev_idx")
            # Desugar `for <target> in reversed(inner)` to
            # `for __rev_idx in range(len(inner)-1, -1, -1):
            #      <target> = inner[__rev_idx]; <body>`.
            # `inner` is substituted directly twice (cheap when it's a Name,
            # Subscript, or simple call such as the benchmark `range(r)`).
            range_call_inner = ast.Call(
                func=ast.Name(id="range", ctx=ast.Load()),
                args=[
                    ast.BinOp(
                        left=ast.Call(
                            func=ast.Name(id="len", ctx=ast.Load()),
                            args=[inner], keywords=[],
                        ),
                        op=ast.Sub(),
                        right=ast.Constant(value=1),
                    ),
                    ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value=1)),
                    ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value=1)),
                ],
                keywords=[],
            )
            sub_inner = ast.Subscript(
                value=inner,
                slice=ast.Name(id=idx_name, ctx=ast.Load()),
                ctx=ast.Load(),
            )
            new_body = [
                ast.Assign(targets=[node.target], value=sub_inner),
                *node.body,
            ]
            new_for = ast.For(
                target=ast.Name(id=idx_name, ctx=ast.Store()),
                iter=range_call_inner,
                body=new_body,
                orelse=node.orelse,
            )
            ast.copy_location(new_for, node)
            ast.fix_missing_locations(new_for)
            return new_for

        # zip(xs, ys, ...) in iter position.
        if isinstance(it.func, ast.Name) and it.func.id == "zip" and len(it.args) >= 2:
            iters = it.args
            idx_name = self._fresh_name("__zip_idx")
            # Per-iteration bindings: target_i = iters[i][__zip_idx]
            # node.target is Tuple of names matching len(iters), or a
            # single name referencing the tuple of values.
            inner_assigns: list = []
            if isinstance(node.target, ast.Tuple) and len(node.target.elts) == len(iters):
                for tgt, src in zip(node.target.elts, iters):
                    inner_assigns.append(
                        ast.Assign(
                            targets=[tgt],
                            value=ast.Subscript(
                                value=src,
                                slice=ast.Name(id=idx_name, ctx=ast.Load()),
                                ctx=ast.Load(),
                            ),
                        )
                    )
            else:
                # Build a Tuple of subscripts and assign once.
                tuple_value = ast.Tuple(
                    elts=[
                        ast.Subscript(
                            value=src,
                            slice=ast.Name(id=idx_name, ctx=ast.Load()),
                            ctx=ast.Load(),
                        )
                        for src in iters
                    ],
                    ctx=ast.Load(),
                )
                inner_assigns.append(
                    ast.Assign(targets=[node.target], value=tuple_value)
                )
            # Iterate i over range(min(len(xs), len(ys), ...)).
            # For just two iters, we can use len(xs) when they're known
            # equal-length — but to stay safe, build min(...).
            len_calls = [
                ast.Call(
                    func=ast.Name(id="len", ctx=ast.Load()),
                    args=[arg], keywords=[],
                )
                for arg in iters
            ]
            if len(len_calls) == 1:
                bound = len_calls[0]
            else:
                bound = ast.Call(
                    func=ast.Name(id="min", ctx=ast.Load()),
                    args=len_calls, keywords=[],
                )
            range_call = ast.Call(
                func=ast.Name(id="range", ctx=ast.Load()),
                args=[bound], keywords=[],
            )
            new_for = ast.For(
                target=ast.Name(id=idx_name, ctx=ast.Store()),
                iter=range_call,
                body=[*inner_assigns, *node.body],
                orelse=node.orelse,
            )
            ast.copy_location(new_for, node)
            ast.fix_missing_locations(new_for)
            return new_for

        return None

    def _fresh_name(self, prefix: str) -> str:
        if not hasattr(self, "_fresh_counter"):
            self._fresh_counter = 0
        self._fresh_counter += 1
        return f"{prefix}_{self._fresh_counter}"

    def visit_While(self, node: ast.While):
        test_expr = self.visit_expr(node.test)
        return {"__class__": "ASTWhileStatement",
                "test_expr": test_expr,
                "block": self.visit_block(node.body), "orelse": self.visit_block(node.orelse)}

    def visit_Assert(self, node: ast.Assert):
        test = self.visit_expr(node.test)
        return {"__class__": "ASTAssertStatement", "expr": test}

    def visit_Raise(self, node: ast.Raise):
        # `raise X(msg)` lowers to `assert False` so mock raises and the
        # constraint system emits unsatisfiable. The path condition (already
        # tracked by visit_assert) ensures the failure only fires on paths
        # that would have actually raised.
        false_const = {"__class__": "ASTConstantBoolean", "value": False}
        return {"__class__": "ASTAssertStatement", "expr": false_const}

    def visit_Pass(self, node: ast.Pass):
        return {"__class__": "ASTPassStatement"}

    def visit_If(self, node: ast.If):
        test = self.visit_expr(node.test)
        return {"__class__": "ASTCondStatement",
                "cond": test, "t_block": self.visit_block(node.body), "f_block": self.visit_block(node.orelse)}

    def visit_Assign(self, node: ast.Assign):
        expr = self.visit_expr(node.value)
        parsed_targets = []
        for target in node.targets:
            parsed_targets.append(self.visit_assign_target(target))
        return {"__class__": "ASTAssignStatement", "targets": parsed_targets, "value": expr}

    def visit_AugAssign(self, node: ast.AugAssign):
        dbg_info = self.get_dbg(node)
        op_type = self.get_op_name_from_node(node.op)
        if isinstance(node.target, ast.Name):
            identifier_name = node.target.id
            expr = self.visit_expr(node.value)
            lhs = {"__class__": "ASTLoad", "name": identifier_name}
            bin_op = {"__class__": "ASTBinaryOperator", "operator": op_type, "lhs": lhs, "rhs": expr}
            return {"__class__": "ASTAssignStatement",
                    "targets": [self.visit_assign_target(node.target)], "value": bin_op}
        elif isinstance(node.target, ast.Subscript):
            # Desugar `a[i] op= v` to `a[i] = a[i] op v` so the augmented
            # form routes through the same well-tested code path as the
            # explicit form. Emitting a distinct ASTAugAssignStatement
            # for subscript targets produced stale-neighbour reads
            # (fuzz-finding-v3-aug-assign-subscript).
            rhs = self.visit_expr(node.value)
            # Python evaluates the index of `a[idx] op= v` ONCE. When the
            # index is a non-trivial expression (e.g. a boolean mask), hoist
            # it into a synthetic temp so the desugared read and write share
            # the same index values — identity-based lowerings (mask-read
            # round-trip cancellation) depend on it, and double evaluation
            # also duplicated compile work. Names/constants are pure and
            # cheap; slices and tuple indices are not first-class temp-able
            # expressions, so those keep the duplicate-expression form.
            sl = node.target.slice
            hoistable = not isinstance(
                sl, (ast.Name, ast.Constant, ast.Slice, ast.Tuple, ast.List))
            if hoistable:
                self._aug_idx_counter = getattr(self, "_aug_idx_counter", 0) + 1
                tmp_name = f"__zinnia_aug_idx_{self._aug_idx_counter}"
                idx_assign = {"__class__": "ASTAssignStatement",
                              "targets": [{"__class__": "ASTNameAssignTarget",
                                           "name": tmp_name, "star": False}],
                              "value": self.visit_expr(sl)}
                slicing = {"__class__": "ASTSlice",
                           "data": [{"__class__": "ASTLoad", "name": tmp_name}]}
                lhs = {"__class__": "ASTSubscriptExp",
                       "val": self.visit_expr(node.target.value),
                       "slicing": slicing}
                bin_op = {"__class__": "ASTBinaryOperator",
                          "operator": op_type, "lhs": lhs, "rhs": rhs}
                target = {"__class__": "ASTSubscriptAssignTarget",
                          "target": self.visit_expr(node.target.value),
                          "slicing": slicing, "star": False}
                return [idx_assign,
                        {"__class__": "ASTAssignStatement",
                         "targets": [target], "value": bin_op}]
            lhs = self.visit_expr(node.target)
            bin_op = {"__class__": "ASTBinaryOperator", "operator": op_type, "lhs": lhs, "rhs": rhs}
            return {"__class__": "ASTAssignStatement",
                    "targets": [self.visit_assign_target(node.target)], "value": bin_op}
        raise InvalidAssignStatementException(dbg_info,
                                              "The value to be assigned must be an identifier name or subscript.")

    def visit_AnnAssign(self, node: ast.AnnAssign):
        dbg_info = self.get_dbg(node)
        # A bare annotation such as `y: Integer` carries no value. The
        # annotation itself is otherwise ignored by the compiler (types are
        # inferred), but a value-less annotation cannot become an assignment.
        if node.value is None:
            raise InvalidAssignStatementException(
                dbg_info,
                "A bare annotation like `x: Integer` has no value to assign; "
                "write `x: Integer = <value>` or drop the annotation.",
            )
        if isinstance(node.target, ast.Name):
            expr = self.visit_expr(node.value)
            return {"__class__": "ASTAssignStatement", "targets": [self.visit_assign_target(node.target)], "value": expr}
        raise InvalidAssignStatementException(dbg_info, "The value to be assigned must be an identifier name.")

    def visit_BinOp(self, node: ast.BinOp):
        left = self.visit_expr(node.left)
        right = self.visit_expr(node.right)
        return {"__class__": "ASTBinaryOperator", "operator": self.get_op_name_from_node(node.op),
                "lhs": left, "rhs": right}

    def visit_Compare(self, node: ast.Compare):
        left = self.visit_expr(node.left)
        comparators = [self.visit_expr(com) for com in node.comparators]
        assert len(node.ops) == len(comparators)
        all_operands = [left] + comparators
        compare_expr_list = []
        for i, op in enumerate(node.ops):
            compare_expr_list.append(
                {"__class__": "ASTBinaryOperator", "operator": self.get_comp_op_name_from_node(op),
                 "lhs": all_operands[i], "rhs": all_operands[i + 1]})
        finalized = compare_expr_list[0]
        for expr in compare_expr_list[1:]:
            finalized = {"__class__": "ASTBinaryOperator", "operator": "and", "lhs": finalized, "rhs": expr}
        return finalized

    def visit_BoolOp(self, node: ast.BoolOp):
        dbg_info = self.get_dbg(node)
        values = [self.visit_expr(val) for val in node.values]
        assert len(values) > 1
        if isinstance(node.op, ast.And):
            base_node = values[0]
            for val in values[1:]:
                base_node = {"__class__": "ASTBinaryOperator", "operator": "and", "lhs": base_node, "rhs": val}
            return base_node
        elif isinstance(node.op, ast.Or):
            base_node = values[0]
            for val in values[1:]:
                base_node = {"__class__": "ASTBinaryOperator", "operator": "or", "lhs": base_node, "rhs": val}
            return base_node
        else:
            raise UnsupportedOperatorException(dbg_info,
                                               f"Invalid boolean operator {type(node.op).__name__} in circuit.")

    def visit_UnaryOp(self, node: ast.UnaryOp):
        value = self.visit_expr(node.operand)
        return {"__class__": "ASTUnaryOperator", "operator": self.get_op_name_from_node(node.op), "operand": value}

    def visit_Call(self, node: ast.Call):
        dbg_info = self.get_dbg(node)
        args = [self.visit_expr(arg) for arg in node.args]
        kwargs = {kwarg.arg: self.visit_expr(kwarg.value) for kwarg in node.keywords}
        # numpy ufunc methods: rewrite to a bare call so the dispatcher in
        # the IR generator sees a flat (target, member) pair.
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Attribute)
            and isinstance(node.func.value.value, ast.Name)
            and node.func.value.value.id == "np"
        ):
            binop = node.func.value.attr
            if node.func.attr in ("outer", "accumulate", "reduceat"):
                return {"__class__": "ASTNamedAttribute",
                        "target": "np", "member": f"{binop}_{node.func.attr}",
                        "args": args, "kwargs": kwargs}
            if node.func.attr == "reduce" and binop in ("add", "multiply", "minimum", "maximum", "logical_or", "logical_and"):
                member = {"add":"sum", "multiply":"prod", "minimum":"min", "maximum":"max", "logical_or":"any", "logical_and":"all"}[binop]
                # ufunc reductions all use the same positional signature,
                # unlike np.min/np.max, whose third argument is out.
                if len(args) > 7:
                    raise ValueError("ufunc.reduce: too many positional arguments")
                for key, value in zip(("axis", "dtype", "out", "keepdims", "initial", "where"), args[1:]):
                    if key in kwargs:
                        raise ValueError(f"ufunc.reduce: multiple values for {key}")
                    kwargs[key] = value
                args = args[:1]
                # NumPy ufunc.reduce defaults axis to 0, while np.sum/prod
                # default to axis=None. Preserve ufunc semantics when the
                # call did not spell axis positionally or by keyword.
                if len(args) == 1 and "axis" not in kwargs:
                    kwargs = {**kwargs, "axis": {"__class__": "ASTConstantInteger", "value": 0}}
                return {"__class__": "ASTNamedAttribute",
                        "target": "np", "member": member,
                        "args": args, "kwargs": kwargs}
        if (isinstance(node.func, ast.Attribute) and node.func.attr=="sliding_window_view"
                and ast.unparse(node.func.value)=="np.lib.stride_tricks"):
            return {"__class__":"ASTNamedAttribute", "target":"np", "member":"sliding_window_view", "args":args, "kwargs":kwargs}
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                return {"__class__": "ASTNamedAttribute",
                        "target": node.func.value.id, "member": node.func.attr,
                        "args": args, "kwargs": kwargs}
            else:
                return {"__class__": "ASTExprAttribute",
                        "target": self.visit_expr(node.func.value), "member": node.func.attr,
                        "args": args, "kwargs": kwargs}
        elif isinstance(node.func, ast.Name):
            return {"__class__": "ASTNamedAttribute", "target": None, "member": node.func.id,
                    "args": args, "kwargs": kwargs}
        raise UnsupportedOperatorException(dbg_info,
                                           f"Invalid call function {type(node.func)}. Only a static specified function name is supported here.")

    def visit_Attribute(self, node: ast.Attribute):
        if isinstance(node.value, ast.Name):
            return {"__class__": "ASTNamedAttribute", "target": node.value.id, "member": node.attr}
        # Non-Name target (e.g., `arr[0].real`, `(a+b).imag`): preserve the
        # target expression so the IR generator can dispatch via
        # visit_expr_attr instead of dropping the target.
        return {"__class__": "ASTExprAttribute",
                "target": self.visit_expr(node.value),
                "member": node.attr,
                "args": [], "kwargs": {}}

    def visit_Name(self, node: ast.Name):
        return {"__class__": "ASTLoad", "name": node.id}

    def visit_Constant(self, node: ast.Constant):
        dbg_info = self.get_dbg(node)
        if node.value is None:
            return {"__class__": "ASTConstantNone"}
        if isinstance(node.value, bool):
            return {"__class__": "ASTConstantBoolean", "value": node.value}
        elif isinstance(node.value, int):
            return {"__class__": "ASTConstantInteger", "value": node.value}
        elif isinstance(node.value, float):
            return {"__class__": "ASTConstantFloat", "value": node.value}
        elif isinstance(node.value, complex):
            return {"__class__": "ASTConstantComplex",
                    "real": float(node.value.real), "imag": float(node.value.imag)}
        elif isinstance(node.value, str):
            return {"__class__": "ASTConstantString", "value": node.value}
        raise UnsupportedConstantLiteralException(dbg_info, f"Invalid constant value `{node.value}` in circuit")

    def visit_List(self, node: ast.List):
        parsed_elts = [self.visit_expr(elt) for elt in node.elts]
        return {"__class__": "ASTSquareBrackets", "values": parsed_elts}

    def visit_Tuple(self, node: ast.Tuple):
        parsed_elts = [self.visit_expr(elt) for elt in node.elts]
        return {"__class__": "ASTParenthesis", "values": parsed_elts}

    def visit_Subscript(self, node: ast.Subscript):
        value = self.visit_expr(node.value)
        return {"__class__": "ASTSubscriptExp", "val": value, "slicing": self.visit_slice_key(node.slice)}

    def visit_Break(self, node: ast.Break):
        return {"__class__": "ASTBreakStatement"}

    def visit_Continue(self, node: ast.Continue):
        return {"__class__": "ASTContinueStatement"}

    def visit_Return(self, node: ast.Return):
        result = None
        if node.value is not None:
            result = self.visit_expr(node.value)
        return {"__class__": "ASTReturnStatement", "expr": result}

    def visit_GeneratorExp(self, node: ast.GeneratorExp):
        dbg = self.get_dbg(node)
        elt = self.visit_expr(node.elt)
        generators = []
        for gen in node.generators:
            target = self.visit_assign_target(gen.target)
            iter_expr = self.visit_expr(gen.iter)
            ifs = [self.visit_expr(if_expr) for if_expr in gen.ifs]
            generators.append({"__class__": "ASTGenerator", "target": target, "iter": iter_expr, "ifs": ifs})
        return {"__class__": "ASTGeneratorExp", "elt": elt, "generators": generators, "kind": "tuple"}

    def visit_ListComp(self, node: ast.ListComp):
        dbg = self.get_dbg(node)
        elt = self.visit_expr(node.elt)
        generators = []
        for gen in node.generators:
            target = self.visit_assign_target(gen.target)
            iter_expr = self.visit_expr(gen.iter)
            ifs = [self.visit_expr(if_expr) for if_expr in gen.ifs]
            generators.append({"__class__": "ASTGenerator", "target": target, "iter": iter_expr, "ifs": ifs})
        return {"__class__": "ASTGeneratorExp", "elt": elt, "generators": generators, "kind": "list"}

    def visit_IfExp(self, node: ast.IfExp):
        test = self.visit_expr(node.test)
        body = self.visit_expr(node.body)
        orelse = self.visit_expr(node.orelse)
        return {"__class__": "ASTCondExp", "cond": test, "t_expr": body, "f_expr": orelse}

    def visit_JoinedStr(self, node: ast.JoinedStr):
        values = [self.visit_expr(v) for v in node.values]
        return {"__class__": "ASTJoinedStr", "values": values}

    def visit_FormattedValue(self, node: ast.FormattedValue):
        return {"__class__": "ASTFormattedValue", "value": self.visit_expr(node.value)}

    def visit_Starred(self, node: ast.Starred):
        return {"__class__": "ASTStarredExpr", "value": self.visit_expr(node.value)}

    def visit_block(self, _stmts):
        stmts = []
        for stmt in _stmts:
            dbg_info = self.get_dbg(stmt)
            if isinstance(stmt, ast.AnnAssign):
                parsed_stmt = self.visit_AnnAssign(stmt)
            elif isinstance(stmt, ast.Assign):
                parsed_stmt = self.visit_Assign(stmt)
            elif isinstance(stmt, ast.For):
                parsed_stmt = self.visit_For(stmt)
            elif isinstance(stmt, ast.While):
                parsed_stmt = self.visit_While(stmt)
            elif isinstance(stmt, ast.Assert):
                parsed_stmt = self.visit_Assert(stmt)
            elif isinstance(stmt, ast.Pass):
                parsed_stmt = self.visit_Pass(stmt)
            elif isinstance(stmt, ast.AugAssign):
                parsed_stmt = self.visit_AugAssign(stmt)
            elif isinstance(stmt, ast.If):
                parsed_stmt = self.visit_If(stmt)
            elif isinstance(stmt, ast.Break):
                parsed_stmt = self.visit_Break(stmt)
            elif isinstance(stmt, ast.Continue):
                parsed_stmt = self.visit_Continue(stmt)
            elif isinstance(stmt, ast.Return):
                parsed_stmt = self.visit_Return(stmt)
            elif isinstance(stmt, ast.Expr):
                parsed_stmt = {"__class__": "ASTExprStatement", "expr": self.visit_expr(stmt.value)}
            elif isinstance(stmt, ast.Import) or isinstance(stmt, ast.ImportFrom):
                # In-circuit imports are runtime no-ops once the module-level
                # Python function has loaded. Keep them so generated benchmark
                # snippets like `import copy; copy.deepcopy(x)` can lower to
                # their supported call semantics.
                parsed_stmt = {"__class__": "ASTPassStatement"}
            elif isinstance(stmt, ast.Raise):
                parsed_stmt = self.visit_Raise(stmt)
            elif isinstance(stmt, ast.Try) or (
                hasattr(ast, "TryStar") and isinstance(stmt, ast.TryStar)
            ):
                # `try` / `except` (including `try/except/else`, `try/finally`,
                # and PEP 654 `try/except*`) has no clean ZK lowering:
                # exception control flow in the constraint system would just
                # be an unsatisfiable branch with no recovery path, and the
                # implicit nature of exception handling makes it easy to
                # introduce length-hiding / selector-privacy leaks by
                # accident. Reject at transform time and point the user at an
                # explicit `if` guard or out-of-circuit validation.
                raise InvalidCircuitStatementException(
                    dbg_info,
                    "`try` / `except` is not supported inside `@zk_circuit`. "
                    "Use an `if` guard for the recoverable case "
                    "(`if <condition>: <fallback> else: <body>`), "
                    "or move the validation outside the circuit."
                )
            elif isinstance(stmt, ast.FunctionDef):
                # Pure inner defs are auto-lifted to chips by autolift_nested_defs
                # before the transformer sees them. If we still see one here,
                # it's either a closure (captures outer-function locals) or
                # otherwise not safe to auto-lift.
                hint = (
                    "It captures variables from the outer function. Move it to module level and pass any captured values as arguments, decorating it with @zk_chip."
                )
                raise InvalidCircuitStatementException(
                    dbg_info,
                    f"nested `def {stmt.name}` cannot be defined inside a circuit body. {hint}"
                )
            else:
                raise InvalidCircuitStatementException(dbg_info, f"Invalid circuit statement defined: {type(stmt)}.")
            if isinstance(parsed_stmt, list):
                stmts.extend(parsed_stmt)
            else:
                stmts.append(parsed_stmt)
        return stmts

    def visit_expr(self, node):
        if isinstance(node, ast.BinOp):
            return self.visit_BinOp(node)
        elif isinstance(node, ast.UnaryOp):
            return self.visit_UnaryOp(node)
        elif isinstance(node, ast.Call):
            return self.visit_Call(node)
        elif isinstance(node, ast.Attribute):
            return self.visit_Attribute(node)
        elif isinstance(node, ast.Name):
            return self.visit_Name(node)
        elif isinstance(node, ast.Constant):
            return self.visit_Constant(node)
        elif isinstance(node, ast.Subscript):
            return self.visit_Subscript(node)
        elif isinstance(node, ast.Compare):
            return self.visit_Compare(node)
        elif isinstance(node, ast.BoolOp):
            return self.visit_BoolOp(node)
        elif isinstance(node, ast.List):
            return self.visit_List(node)
        elif isinstance(node, ast.Tuple):
            return self.visit_Tuple(node)
        elif isinstance(node, ast.GeneratorExp):
            return self.visit_GeneratorExp(node)
        elif isinstance(node, ast.ListComp):
            return self.visit_ListComp(node)
        elif isinstance(node, ast.IfExp):
            return self.visit_IfExp(node)
        elif isinstance(node, ast.JoinedStr):
            return self.visit_JoinedStr(node)
        elif isinstance(node, ast.FormattedValue):
            return self.visit_FormattedValue(node)
        elif isinstance(node, ast.Starred):
            return self.visit_Starred(node)
        elif isinstance(node, ast.Lambda):
            dbg_info = self.get_dbg(node)
            raise UnsupportedLangFeatureException(
                dbg_info,
                "Lambda expressions are not supported inside @zk_circuit. "
                "Define a named helper function or inline the expression instead."
            )
        elif isinstance(node, ast.Set):
            dbg_info = self.get_dbg(node)
            raise UnsupportedLangFeatureException(
                dbg_info,
                "Set literals are not supported inside @zk_circuit because circuits have no set type. "
                "Use a list or tuple for finite membership checks instead."
            )
        elif isinstance(node, ast.Dict):
            dbg_info = self.get_dbg(node)
            raise UnsupportedLangFeatureException(
                dbg_info,
                "Dict literals are not supported inside @zk_circuit because mapping types are unsupported in circuits."
            )
        else:
            dbg_info = self.get_dbg(node)
            raise UnsupportedLangFeatureException(dbg_info, f"Expression transformation rule for {type(node)} is not implemented.")

    def visit_annotation(self, node, name: str | None):
        """Parse a type annotation and return {"kind": str|None, "dt": full_type_dict}.

        The full_type_dict has format {"__class__": "XxxDTDescriptor", "dt_data": {...}}.
        """
        kind = None
        error_msg = f"Invalid annotation for `{name}`." if name is not None else "Invalid annotation."
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            if node.value.id == "Public":
                kind = "Public"
                node = node.slice
            elif node.value.id == "Private":
                kind = "Private"
                node = node.slice

        def _parse_literal_shape_string(dbg, shape_value: str, marker_name: str) -> list[int]:
            if not isinstance(shape_value, str):
                raise InvalidAnnotationException(
                    dbg,
                    error_msg + " Dtype-front tensor annotations require a string shape, e.g. `Float[Array, \"2 3\"]`."
                )
            dims = shape_value.split()
            if not dims:
                raise InvalidAnnotationException(dbg, error_msg + " Tensor shape string must not be empty.")
            parsed = []
            for dim in dims:
                if dim.isdecimal():
                    value = int(dim)
                    if value <= 0:
                        raise InvalidAnnotationException(dbg, error_msg + " Tensor shape dimensions must be positive integers.")
                    parsed.append(value)
                    continue
                if marker_name == "Array" and "=" in dim:
                    name, value_text = dim.split("=", 1)
                    if name.isidentifier() and value_text.isdecimal():
                        value = int(value_text)
                        if value <= 0:
                            raise InvalidAnnotationException(
                                dbg,
                                error_msg + " Tensor shape dimensions must be positive integers."
                            )
                        parsed.append((name, value))
                        continue
                if marker_name == "Array" and "<=" in dim:
                    name, bound_text = dim.split("<=", 1)
                    if name.isidentifier():
                        try:
                            bound = _eval_shape_expr(_parse_shape_expr_token(bound_text), self.shape_symbol_bindings)
                        except ValueError as exc:
                            raise InvalidAnnotationException(dbg, error_msg + f" {exc}") from exc
                        if bound is None:
                            raise InvalidAnnotationException(
                                dbg,
                                error_msg + f" Bounded shape expression `{dim}` could not be reduced to a public upper bound."
                            )
                        if bound <= 0:
                            raise InvalidAnnotationException(
                                dbg,
                                error_msg + " Tensor shape upper bounds must be positive integers."
                            )
                        self.shape_symbol_names.add(name)
                        self.shape_symbol_bounds[name] = bound
                        parsed.append(("bound", name, bound))
                        continue
                if marker_name == "Array":
                    try:
                        expr = _parse_shape_expr_token(dim)
                    except ValueError as exc:
                        raise InvalidAnnotationException(dbg, error_msg + f" {exc}") from exc
                    value = _eval_shape_expr(expr, self.shape_symbol_bindings)
                    if value is not None:
                        if value <= 0:
                            raise InvalidAnnotationException(
                                dbg,
                                error_msg + " Tensor shape dimensions must be positive integers."
                        )
                        parsed.append(value)
                        continue
                    if isinstance(expr, tuple) and len(expr) == 2 and expr[0] == "symbol":
                        self.shape_symbol_names.add(expr[1])
                        parsed.append(expr[1])
                        continue
                    for symbol in _shape_expr_symbols(expr):
                        self.shape_symbol_names.add(symbol)
                    parsed.append(_shape_expr_to_string(expr))
                    continue
                raise InvalidAnnotationException(
                    dbg,
                    error_msg + f" `{marker_name}` annotations are recognized, but DynArray/Vec and bounded symbolic shapes are deferred in the literal-static v1 slice."
                )
            return parsed

        def _parse_dynarray_bound_string(dbg, shape_value: str) -> tuple[int, int]:
            if not isinstance(shape_value, str):
                raise InvalidAnnotationException(
                    dbg,
                    error_msg + " DynArray annotations require a bound string, e.g. `Float[DynArray, \"N<=1024\"]`."
                )
            dims = shape_value.split()
            if len(dims) != 1:
                raise InvalidAnnotationException(
                    dbg,
                    error_msg + " DynArray input annotations currently support one bounded dimension, e.g. `Float[DynArray, \"N<=1024\"]`."
                )
            dim = dims[0]
            if "<=" not in dim:
                if dim.isidentifier():
                    self.shape_symbol_names.add(dim)
                    return ("symbol", dim), 1
                raise InvalidAnnotationException(
                    dbg,
                    error_msg + " DynArray dimensions must provide a symbol or an upper bound, e.g. `N` or `N<=1024`."
                )
            name, bound_text = dim.split("<=", 1)
            if not name.isidentifier():
                raise InvalidAnnotationException(
                    dbg,
                    error_msg + " DynArray bounds must have the form `Name<=positive_bound`."
                )
            try:
                bound = _eval_shape_expr(_parse_shape_expr_token(bound_text), self.shape_symbol_bindings)
            except ValueError as exc:
                raise InvalidAnnotationException(dbg, error_msg + f" {exc}") from exc
            if bound is None:
                raise InvalidAnnotationException(
                    dbg,
                    error_msg + f" DynArray bound expression `{bound_text}` could not be reduced to a public upper bound."
                )
            if bound <= 0:
                raise InvalidAnnotationException(
                    dbg,
                    error_msg + " DynArray upper bounds must be positive integers."
                )
            self.shape_symbol_names.add(name)
            self.shape_symbol_bounds[name] = bound
            return bound, 1

        def _parse_dtype_front_tensor(_n):
            if (
                not isinstance(_n, ast.Subscript)
                or not isinstance(_n.value, ast.Name)
                or _n.value.id not in _DTYPE_FRONT_ALIASES
            ):
                return None
            if not isinstance(_n.slice, ast.Tuple) or len(_n.slice.elts) != 2:
                raise InvalidAnnotationException(
                    self.get_dbg(_n),
                    error_msg + " Dtype-front tensor annotations use `DType[Array, \"shape\"]`."
                )
            marker_node, shape_node = _n.slice.elts
            if not isinstance(marker_node, ast.Name):
                raise InvalidAnnotationException(self.get_dbg(_n), error_msg)
            marker_name = marker_node.id
            if marker_name == "DynArray":
                if not isinstance(shape_node, ast.Constant):
                    raise InvalidAnnotationException(self.get_dbg(shape_node), error_msg)
                max_length, max_rank = _parse_dynarray_bound_string(
                    self.get_dbg(shape_node), shape_node.value
                )
                dtype = make_type_dict(self.get_dbg(_n.value), _DTYPE_FRONT_ALIASES[_n.value.id])
                if isinstance(max_length, tuple) and max_length[0] == "symbol":
                    return make_type_dict(
                        self.get_dbg(_n),
                        "SymbolicNDArray",
                        tuple([dtype, max_length[1]]),
                    )
                return make_type_dict(
                    self.get_dbg(_n),
                    "DynamicNDArray",
                    tuple([dtype, max_length, max_rank]),
                )
            if marker_name != "Array":
                raise InvalidAnnotationException(
                    self.get_dbg(marker_node),
                    error_msg + f" `{marker_name}` annotations are recognized, but DynArray/Vec and bounded symbolic shapes are deferred in the literal-static v1 slice."
                )
            if not isinstance(shape_node, ast.Constant):
                raise InvalidAnnotationException(self.get_dbg(shape_node), error_msg)
            shape = _parse_literal_shape_string(self.get_dbg(shape_node), shape_node.value, marker_name)
            dtype = make_type_dict(self.get_dbg(_n.value), _DTYPE_FRONT_ALIASES[_n.value.id])
            typename = "SymbolicNDArray" if any(not isinstance(d, int) for d in shape) else "NDArray"
            return make_type_dict(self.get_dbg(_n), typename, tuple([dtype, *shape]))

        def _reject_deferred_vec(_n):
            if isinstance(_n, ast.Subscript) and isinstance(_n.value, ast.Name) and _n.value.id == "Vec":
                raise InvalidAnnotationException(
                    self.get_dbg(_n),
                    error_msg + " `Vec[...]` annotations are recognized, but sized homogeneous list lowering is deferred in the literal-static v1 slice."
                )

        def _resolve_symbolic_bindings(dt):
            if not isinstance(dt, dict) or "SymbolicNDArray" not in dt:
                return dt
            data = dict(dt["SymbolicNDArray"])
            bindings = dict(data.get("bindings") or {})
            for dim in data["shape"]:
                if isinstance(dim, str):
                    self.shape_symbol_names.update(_shape_expr_symbols(_parse_shape_expr_token(dim)))
            for dim in data["shape"]:
                if isinstance(dim, str) and dim not in bindings and dim in self.shape_symbol_bindings:
                    bindings[dim] = self.shape_symbol_bindings[dim]
            for symbol, value in bindings.items():
                known = self.shape_symbol_bindings.get(symbol)
                if known is not None and known != value:
                    raise InvalidAnnotationException(
                        self.get_dbg(node),
                        error_msg + f" Conflicting exact shape bindings for `{symbol}`: {known} vs {value}."
                    )
                self.shape_symbol_bindings[symbol] = value
            data["bindings"] = bindings
            return {"SymbolicNDArray": data}

        def _inner_parser(_n):
            dtype_front = _parse_dtype_front_tensor(_n)
            if dtype_front is not None:
                return dtype_front
            _reject_deferred_vec(_n)
            if isinstance(_n, ast.Name):
                return make_type_dict(self.get_dbg(_n), _n.id)
            elif isinstance(_n, ast.Subscript):
                if not isinstance(_n.value, ast.Name):
                    raise InvalidAnnotationException(self.get_dbg(_n), error_msg)
                if isinstance(_n.slice, ast.Tuple):
                    args = []
                    for elt in _n.slice.elts:
                        if isinstance(elt, ast.Name):
                            args.append(_inner_parser(elt))
                        elif isinstance(elt, ast.Subscript):
                            args.append(_inner_parser(elt))
                        elif isinstance(elt, ast.Constant):
                            args.append(elt.value)
                        elif isinstance(elt, ast.Tuple):
                            if not all(isinstance(e, ast.Constant) for e in elt.elts):
                                raise InvalidAnnotationException(
                                    self.get_dbg(elt), error_msg + f" All tuple elements should be constant.")
                            args.append(tuple(e.value for e in elt.elts))
                        else:
                            raise InvalidAnnotationException(self.get_dbg(_n), error_msg)
                    return make_type_dict(self.get_dbg(_n), _n.value.id, tuple(args))
                elif isinstance(_n.slice, ast.Constant):
                    return make_type_dict(self.get_dbg(_n), _n.value.id, (_n.slice.value,))
                elif isinstance(_n.slice, ast.Name):
                    return make_type_dict(self.get_dbg(_n), _n.value.id, (_inner_parser(_n.slice),))
                elif isinstance(_n.slice, ast.Subscript):
                    return make_type_dict(self.get_dbg(_n), _n.value.id, (_inner_parser(_n.slice),))
            elif isinstance(_n, ast.Constant):
                if _n.value is None:
                    return make_type_dict(self.get_dbg(_n), "None")
                raise InvalidAnnotationException(
                    self.get_dbg(_n), error_msg + f" Constant value {_n.value} is not supported as an annotation.")
            raise InvalidAnnotationException(self.get_dbg(_n), error_msg)

        return {"kind": kind, "dt": _resolve_symbolic_bindings(_inner_parser(node))}

    def visit_slice_key(self, node):
        constant_none = {"__class__": "ASTConstantNone"}
        ellipsis_sentinel = {"__class__": "ASTSliceEllipsis"}
        newaxis_sentinel = {"__class__": "ASTSliceNewAxis"}

        def _slice_key_elt(elt):
            # `...` -> ellipsis sentinel.
            if isinstance(elt, ast.Constant) and elt.value is Ellipsis:
                return ellipsis_sentinel
            # `None` literal -> np.newaxis.
            if isinstance(elt, ast.Constant) and elt.value is None:
                return newaxis_sentinel
            # `np.newaxis` -> np.newaxis. We accept any attribute access whose
            # attribute name is `newaxis`; resolving the actual numpy module is
            # not worth the trouble for what is, by convention, a sentinel.
            if isinstance(elt, ast.Attribute) and elt.attr == "newaxis":
                return newaxis_sentinel
            return self.visit_expr(elt)

        if isinstance(node, ast.Slice):
            lo = self.visit_expr(node.lower) if node.lower is not None else constant_none
            hi = self.visit_expr(node.upper) if node.upper is not None else constant_none
            step = self.visit_expr(node.step) if node.step is not None else constant_none
            return {"__class__": "ASTSlice", "data": [[lo, hi, step]]}
        elif isinstance(node, ast.Tuple):
            slicing_data = []
            for elt in node.elts:
                if isinstance(elt, ast.Slice):
                    lo = self.visit_expr(elt.lower) if elt.lower is not None else constant_none
                    hi = self.visit_expr(elt.upper) if elt.upper is not None else constant_none
                    step = self.visit_expr(elt.step) if elt.step is not None else constant_none
                    slicing_data.append([lo, hi, step])
                else:
                    slicing_data.append(_slice_key_elt(elt))
            return {"__class__": "ASTSlice", "data": slicing_data}
        return {"__class__": "ASTSlice", "data": [_slice_key_elt(node)]}

    def visit_assign_target(self, node, starred=False):
        if isinstance(node, ast.Name):
            return {"__class__": "ASTNameAssignTarget", "name": node.id, "star": starred}
        elif isinstance(node, ast.Subscript):
            return {"__class__": "ASTSubscriptAssignTarget",
                    "target": self.visit_expr(node.value),
                    "slicing": self.visit_slice_key(node.slice), "star": starred}
        elif isinstance(node, ast.Tuple):
            elements = [self.visit_assign_target(elt) for elt in node.elts]
            return {"__class__": "ASTTupleAssignTarget", "targets": elements, "star": starred}
        elif isinstance(node, ast.List):
            elements = [self.visit_assign_target(elt) for elt in node.elts]
            return {"__class__": "ASTListAssignTarget", "targets": elements, "star": starred}
        elif isinstance(node, ast.Starred):
            return self.visit_assign_target(node.value, True)
        raise InvalidAssignStatementException(
            self.get_dbg(node),
            f"Unsupported assignment target `{type(node).__name__}`; "
            f"only names, subscripts, and (nested) tuple/list targets are supported.",
        )
