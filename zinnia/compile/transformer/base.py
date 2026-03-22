import ast

from zinnia.debug.exception import InvalidCircuitStatementException, \
    InvalidProgramException, InvalidAssignStatementException, InvalidAnnotationException, UnsupportedOperatorException, \
    UnsupportedConstantLiteralException, \
    UnsupportedLangFeatureException
from zinnia.debug.dbg_info import DebugInfo


# Type name to DTDescriptor class name mapping (matches DTDescriptorFactory.DATATYPE_REGISTRY order)
_TYPE_ALIASES = {
    "DynamicNDArray": "DynamicNDArrayDTDescriptor",
    "NDArray": "NDArrayDTDescriptor",
    "Tuple": "TupleDTDescriptor", "tuple": "TupleDTDescriptor",
    "List": "ListDTDescriptor", "list": "ListDTDescriptor",
    "Integer": "IntegerDTDescriptor", "int": "IntegerDTDescriptor", "Int": "IntegerDTDescriptor", "integer": "IntegerDTDescriptor",
    "Boolean": "IntegerDTDescriptor", "bool": "IntegerDTDescriptor", "Bool": "IntegerDTDescriptor", "boolean": "IntegerDTDescriptor",
    "Float": "FloatDTDescriptor", "float": "FloatDTDescriptor",
    "None": "NoneDTDescriptor",
    "Class": "ClassDTDescriptor",
    "PoseidonHashed": "PoseidonHashedDTDescriptor",
    "String": "StringDTDescriptor", "str": "StringDTDescriptor",
}


def make_type_dict(dbg, typename, args=None):
    """Create a type descriptor dict matching the format Rust expects.

    Returns: {"__class__": "<ClassName>", "dt_data": {...}}
    """
    if args is None:
        args = ()

    class_name = _TYPE_ALIASES.get(typename)
    if class_name is None:
        raise InvalidAnnotationException(dbg, f'`{typename}` is not a valid type name')

    # Simple scalar types (no args needed)
    if class_name in ("IntegerDTDescriptor", "FloatDTDescriptor", "NoneDTDescriptor",
                       "StringDTDescriptor", "ClassDTDescriptor"):
        return {"__class__": class_name, "dt_data": {}}

    # NDArray: args = (dtype_dict, *shape_ints)
    if class_name == "NDArrayDTDescriptor":
        if len(args) < 2:
            raise InvalidAnnotationException(dbg, f"NDArray requires at least 2 arguments (dtype and shape dimensions).")
        dtype_dict = args[0]
        if not isinstance(dtype_dict, dict):
            raise InvalidAnnotationException(dbg, f"First argument to NDArray must be a type.")
        shape = []
        for a in args[1:]:
            if not isinstance(a, int):
                raise InvalidAnnotationException(dbg, f"NDArray shape dimensions must be integers.")
            shape.append(a)
        return {"__class__": "NDArrayDTDescriptor", "dt_data": {"dtype": dtype_dict, "shape": shape}}

    # DynamicNDArray: args = (dtype_dict, max_length, max_rank)
    if class_name == "DynamicNDArrayDTDescriptor":
        if len(args) != 3:
            raise InvalidAnnotationException(dbg, f"DynamicNDArray requires exactly 3 arguments (dtype, max_length, max_rank).")
        dtype_dict = args[0]
        if not isinstance(dtype_dict, dict):
            raise InvalidAnnotationException(dbg, f"First argument to DynamicNDArray must be a type.")
        if not isinstance(args[1], int) or not isinstance(args[2], int):
            raise InvalidAnnotationException(dbg, f"DynamicNDArray max_length and max_rank must be integers.")
        return {"__class__": "DynamicNDArrayDTDescriptor", "dt_data": {
            "dtype": dtype_dict, "max_length": args[1], "max_rank": args[2]
        }}

    # List: args = (element_type_dicts...)
    if class_name == "ListDTDescriptor":
        if len(args) < 1:
            raise InvalidAnnotationException(dbg, f"List requires at least 1 argument.")
        elements = []
        for a in args:
            if not isinstance(a, dict):
                raise InvalidAnnotationException(dbg, f"List element types must be type descriptors.")
            elements.append(a)
        return {"__class__": "ListDTDescriptor", "dt_data": {"elements": elements}}

    # Tuple: args = (element_type_dicts_or_ints...)
    if class_name == "TupleDTDescriptor":
        if len(args) < 1:
            raise InvalidAnnotationException(dbg, f"Tuple requires at least 1 argument.")
        elements = list(args)
        return {"__class__": "TupleDTDescriptor", "dt_data": {"elements": elements}}

    # PoseidonHashed: args = (dtype_dict,)
    if class_name == "PoseidonHashedDTDescriptor":
        if len(args) != 1:
            raise InvalidAnnotationException(dbg, f"PoseidonHashed requires exactly 1 argument.")
        if not isinstance(args[0], dict):
            raise InvalidAnnotationException(dbg, f"PoseidonHashed argument must be a type descriptor.")
        return {"__class__": "PoseidonHashedDTDescriptor", "dt_data": {"dtype": args[0]}}

    raise InvalidAnnotationException(dbg, f'`{typename}` is not a valid type name')


class ZinniaBaseASTTransformer(ast.NodeTransformer):
    def __init__(self, source_code: str, method_name: str):
        super().__init__()
        self.source_code = source_code
        self.method_name = method_name

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
        raise UnsupportedOperatorException(self.get_dbg(node), f"Invalid operator {type(node.op).__name__} in circuit.")

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
        raise UnsupportedOperatorException(self.get_dbg(node),
                                           f"Invalid compare operator {type(node).__name__} in circuit.")

    def get_dbg(self, node) -> DebugInfo:
        raise NotImplementedError()

    def visit(self, node):
        if isinstance(node, ast.FunctionDef):
            return self.visit_FunctionDef(node)
        raise InvalidProgramException(None, "Invalid code passed to the compiler! The program must be a function.")

    def visit_For(self, node: ast.For):
        target = self.visit_assign_target(node.target)
        iter_expr = self.visit_expr(node.iter)
        return {"__class__": "ASTForInStatement",
                "target": target, "iter_expr": iter_expr,
                "block": self.visit_block(node.body), "orelse": self.visit_block(node.orelse)}

    def visit_While(self, node: ast.While):
        test_expr = self.visit_expr(node.test)
        return {"__class__": "ASTWhileStatement",
                "test_expr": test_expr,
                "block": self.visit_block(node.body), "orelse": self.visit_block(node.orelse)}

    def visit_Assert(self, node: ast.Assert):
        test = self.visit_expr(node.test)
        return {"__class__": "ASTAssertStatement", "expr": test}

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
            expr = self.visit_expr(node.value)
            return {"__class__": "ASTAugAssignStatement",
                    "targets": [self.visit_assign_target(node.target)], "value": expr, "op_type": op_type}
        raise InvalidAssignStatementException(dbg_info,
                                              "The value to be assigned must be an identifier name or subscript.")

    def visit_AnnAssign(self, node: ast.AnnAssign):
        dbg_info = self.get_dbg(node)
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
        return {"__class__": "ASTNamedAttribute", "target": None, "member": node.attr}

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
            else:
                raise InvalidCircuitStatementException(dbg_info, f"Invalid circuit statement defined: {type(stmt)}.")
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

        def _inner_parser(_n):
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

        return {"kind": kind, "dt": _inner_parser(node)}

    def visit_slice_key(self, node):
        constant_none = {"__class__": "ASTConstantNone"}
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
                    slicing_data.append(self.visit_expr(elt))
            return {"__class__": "ASTSlice", "data": slicing_data}
        return {"__class__": "ASTSlice", "data": [self.visit_expr(node)]}

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
        raise NotImplementedError()
