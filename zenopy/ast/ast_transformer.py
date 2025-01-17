import ast
from typing import Dict

from zenopy.debug.exception import InvalidCircuitInputException, InvalidCircuitStatementException, \
    InvalidProgramException, InvalidAssignStatementException, InvalidAnnotationException, InvalidSlicingException, \
    UnsupportedOperatorException, UnsupportedConstantLiteralException, \
    UnsupportedLangFeatureException, InvalidForStatementException, InvalidChipInputException
from zenopy.ast.zk_ast import ASTProgramInput, ASTAnnotation, ASTProgram, ASTAssignStatement, \
    ASTLoad, ASTSlicingAssignStatement, ASTForInStatement, ASTPassStatement, ASTAssertStatement, ASTCondStatement, \
    ASTSlicing, ASTSquareBrackets, ASTBreakStatement, ASTContinueStatement, ASTBinaryOperator, ASTUnaryOperator, \
    ASTNamedAttribute, ASTExprAttribute, ASTParenthesis, ASTChip, ASTChipInput, ASTReturnStatement, ASTCallStatement, \
    ASTConstantInteger, ASTConstantFloat, ASTSlice, ASTConstantNone
from zenopy.internal.chip_object import ChipObject
from zenopy.internal.dt_descriptor import DTDescriptorFactory, NoneDTDescriptor
from zenopy.internal.input_anno_name import InputAnnoName
from zenopy.debug.dbg_info import DebugInfo


class PyZKBaseASTTransformer(ast.NodeTransformer):
    def __init__(self, source_code: str, method_name: str):
        super().__init__()
        self.source_code = source_code
        self.method_name = method_name

    def get_debug_info(self, node) -> DebugInfo:
        raise NotImplementedError()

    def visit(self, node):
        if isinstance(node, ast.FunctionDef):
            return self.visit_FunctionDef(node)
        raise InvalidProgramException(None, "Invalid program passed to the compiler! The program must be a function.")

    def visit_For(self, node):
        if not isinstance(node.target, ast.Name):
            target_pos_info = self.get_debug_info(node.target)
            raise InvalidForStatementException(target_pos_info, 'In for statement, the variable before keyword "in" must be a name.')
        target_name = node.target.id
        iter_expr = self.visit_expr(node.iter)
        return ASTForInStatement(self.get_debug_info(node.iter), target_name, iter_expr, self.visit_block(node.body))

    def visit_Assert(self, node):
        dbg_info = self.get_debug_info(node)
        test = self.visit_expr(node.test)
        return ASTAssertStatement(dbg_info, test)

    def visit_Pass(self, node):
        dbg_info = self.get_debug_info(node)
        return ASTPassStatement(dbg_info)

    def visit_If(self, node):
        dbg_info = self.get_debug_info(node)
        test = self.visit_expr(node.test)
        return ASTCondStatement(dbg_info, test, self.visit_block(node.body), self.visit_block(node.orelse))

    def visit_Assign(self, node):
        dbg_info = self.get_debug_info(node)
        if len(node.targets) != 1:
            raise InvalidAssignStatementException(dbg_info, "The assignment statement does not support multiple assignments (unpacking).")
        if isinstance(node.targets[0], ast.Name):
            identifier_name = node.targets[0].id
            expr = self.visit_expr(node.value)
            return ASTAssignStatement(dbg_info, identifier_name, expr, None)
        elif isinstance(node.targets[0], ast.Subscript):
            assignee = self.visit_expr(node.targets[0].value)
            expr = self.visit_expr(node.value)
            return ASTSlicingAssignStatement(dbg_info, assignee, self.visit_slice_key(node.targets[0].slice), expr)
        raise InvalidAssignStatementException(dbg_info, "The value to be assigned must be an identifier name or name with slicing.")

    def visit_AugAssign(self, node):
        dbg_info = self.get_debug_info(node)
        if isinstance(node.op, ast.Add):
            op_type = ASTBinaryOperator.Op.ADD
        elif isinstance(node.op, ast.Sub):
            op_type = ASTBinaryOperator.Op.SUB
        elif isinstance(node.op, ast.Div):
            op_type = ASTBinaryOperator.Op.DIV
        elif isinstance(node.op, ast.Mult):
            op_type = ASTBinaryOperator.Op.MUL
        elif isinstance(node.op, ast.MatMult):
            op_type = ASTBinaryOperator.Op.MAT_MUL
        elif isinstance(node.op, ast.FloorDiv):
            op_type = ASTBinaryOperator.Op.FLOOR_DIV
        elif isinstance(node.op, ast.Mod):
            op_type = ASTBinaryOperator.Op.MOD
        else:
            raise InvalidAssignStatementException(dbg_info, f"Invalid augmented assignment operator {type(node.op)}")
        if isinstance(node.target, ast.Name):
            identifier_name = node.target.id
            expr = self.visit_expr(node.value)
            return ASTAssignStatement(
                dbg_info, identifier_name, ASTBinaryOperator(
                    dbg_info, op_type, ASTLoad(self.get_debug_info(node.target), identifier_name), expr
                ), None)
        elif isinstance(node.target, ast.Subscript):
            assignee = self.visit_expr(node.target.value)
            expr = self.visit_expr(node.value)
            return ASTSlicingAssignStatement(dbg_info, assignee, self.visit_slice_key(node.target.slice), ASTBinaryOperator(
                    self.get_debug_info(node.value), op_type, assignee, expr))
        raise InvalidAssignStatementException(dbg_info, "The value to be assigned must be an identifier name or name with slicing.")

    def visit_AnnAssign(self, node):
        dbg_info = self.get_debug_info(node)
        if not isinstance(node.target, ast.Name):
            raise InvalidAssignStatementException(dbg_info, "The value to be assigned must be an identifier name.")
        if isinstance(node.target, ast.Name):
            identifier_name = node.target.id
            expr = self.visit_expr(node.value)
            annotation = self.visit_annotation(node.annotation, identifier_name)
            return ASTAssignStatement(dbg_info, identifier_name, expr, annotation)
        elif isinstance(node.target, ast.Subscript):
            assignee = self.visit_expr(node.target.value)
            expr = self.visit_expr(node.value)
            return ASTSlicingAssignStatement(dbg_info, assignee, self.visit_slice_key(node.target.slice), expr)
        raise InvalidAssignStatementException(dbg_info, "The value to be assigned must be an identifier name or name with slicing.")

    def visit_BinOp(self, node):
        dbg_info = self.get_debug_info(node)
        left = self.visit_expr(node.left)
        right = self.visit_expr(node.right)
        if isinstance(node.op, ast.Add):
            return ASTBinaryOperator(dbg_info, ASTBinaryOperator.Op.ADD, left, right)
        elif isinstance(node.op, ast.Mult):
            return ASTBinaryOperator(dbg_info, ASTBinaryOperator.Op.MUL, left, right)
        elif isinstance(node.op, ast.MatMult):
            return ASTBinaryOperator(dbg_info, ASTBinaryOperator.Op.MAT_MUL, left, right)
        elif isinstance(node.op, ast.Div):
            return ASTBinaryOperator(dbg_info, ASTBinaryOperator.Op.DIV, left, right)
        elif isinstance(node.op, ast.Sub):
            return ASTBinaryOperator(dbg_info, ASTBinaryOperator.Op.SUB, left, right)
        elif isinstance(node.op, ast.Pow):
            return ASTBinaryOperator(dbg_info, ASTBinaryOperator.Op.POW, left, right)
        elif isinstance(node.op, ast.FloorDiv):
            return ASTBinaryOperator(dbg_info, ASTBinaryOperator.Op.FLOOR_DIV, left, right)
        elif isinstance(node.op, ast.Mod):
            return ASTBinaryOperator(dbg_info, ASTBinaryOperator.Op.MOD, left, right)
        else:
            raise UnsupportedOperatorException(dbg_info, f"Invalid binary operator {type(node.op).__name__} in circuit.")

    def visit_Compare(self, node):
        dbg_info = self.get_debug_info(node)
        left = self.visit_expr(node.left)
        comparators = []
        for com in node.comparators:
            comparators.append(self.visit_expr(com))
        assert len(node.ops) == len(comparators)
        comparators = [left] + comparators
        compare_expr_list = []
        for i, op in enumerate(node.ops):
            if isinstance(op, ast.GtE):
                compare_expr_list.append(ASTBinaryOperator(dbg_info, ASTBinaryOperator.Op.GTE, comparators[i], comparators[i + 1]))
            elif isinstance(op, ast.LtE):
                compare_expr_list.append(ASTBinaryOperator(dbg_info, ASTBinaryOperator.Op.LTE, comparators[i], comparators[i + 1]))
            elif isinstance(op, ast.Gt):
                compare_expr_list.append(ASTBinaryOperator(dbg_info, ASTBinaryOperator.Op.GT, comparators[i], comparators[i + 1]))
            elif isinstance(op, ast.Lt):
                compare_expr_list.append(ASTBinaryOperator(dbg_info, ASTBinaryOperator.Op.LT, comparators[i], comparators[i + 1]))
            elif isinstance(op, ast.Eq):
                compare_expr_list.append(ASTBinaryOperator(dbg_info, ASTBinaryOperator.Op.EQ, comparators[i], comparators[i + 1]))
            elif isinstance(op, ast.NotEq):
                compare_expr_list.append(ASTBinaryOperator(dbg_info, ASTBinaryOperator.Op.NE, comparators[i], comparators[i + 1]))
            else:
                raise UnsupportedOperatorException(dbg_info, f"Invalid compare operator {type(op).__name__} in circuit.")
        finalized = compare_expr_list[0]
        for expr in compare_expr_list[1:]:
            finalized = ASTBinaryOperator(dbg_info, ASTBinaryOperator.Op.AND, finalized, expr)
        return finalized

    def visit_BoolOp(self, node):
        dbg_info = self.get_debug_info(node)
        values = []
        for val in node.values:
            values.append(self.visit_expr(val))
        assert len(values) > 1
        if isinstance(node.op, ast.And):
            base_node = values[0]
            for val in values[1:]:
                base_node = ASTBinaryOperator(dbg_info, ASTBinaryOperator.Op.AND, base_node, val)
            return base_node
        elif isinstance(node.op, ast.Or):
            base_node = values[0]
            for val in values[1:]:
                base_node = ASTBinaryOperator(dbg_info, ASTBinaryOperator.Op.OR, base_node, val)
            return base_node
        else:
            raise UnsupportedOperatorException(dbg_info, f"Invalid boolean operator {type(node.op).__name__} in circuit.")

    def visit_UnaryOp(self, node):
        dbg_info = self.get_debug_info(node)
        value = self.visit_expr(node.operand)
        if isinstance(node.op, ast.Not):
            return ASTUnaryOperator(dbg_info, ASTUnaryOperator.Op.NOT, value)
        elif isinstance(node.op, ast.USub):
            return ASTUnaryOperator(dbg_info, ASTUnaryOperator.Op.USUB, value)
        elif isinstance(node.op, ast.UAdd):
            return value
        else:
            raise UnsupportedOperatorException(dbg_info, f"Invalid unary operator {type(node.op)} in circuit.")

    def visit_Call(self, node):
        dbg_info = self.get_debug_info(node)
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                before_name = node.func.value.id
                after_name = node.func.attr
                return ASTNamedAttribute(
                    dbg_info, before_name, after_name,
                    [self.visit_expr(arg) for arg in node.args],
                    {kwarg.arg: self.visit_expr(kwarg.value) for kwarg in node.keywords}
                )
            else:
                after_name = node.func.attr
                return ASTExprAttribute(
                    dbg_info, self.visit_expr(node.func.value), after_name,
                    [self.visit_expr(arg) for arg in node.args],
                    {kwarg.arg: self.visit_expr(kwarg.value) for kwarg in node.keywords}
                )
        elif isinstance(node.func, ast.Name):
            return ASTNamedAttribute(
                dbg_info, None, node.func.id,
                [self.visit_expr(arg) for arg in node.args],
                {kwarg.arg: self.visit_expr(kwarg.value) for kwarg in node.keywords}
            )
        raise UnsupportedOperatorException(dbg_info, f"Invalid call function {type(node.func)}. Only a static specified function name is supported here.")

    def visit_CallStmt(self, node):
        dbg_info = self.get_debug_info(node)
        if isinstance(node.func, ast.Name):
            return ASTCallStatement(
                dbg_info, node.func.id,
                [self.visit_expr(arg) for arg in node.args],
                {kwarg.arg: self.visit_expr(kwarg.value) for kwarg in node.keywords}
            )
        raise UnsupportedOperatorException(dbg_info, f"Invalid call function {type(node.func)}. Only a static specified function name is supported here.")

    def visit_Attribute(self, node: ast.Attribute):
        dbg_info = self.get_debug_info(node)
        if isinstance(node.value, ast.Name):
            after_name = node.attr
            before_name = node.value.id
            return ASTNamedAttribute(dbg_info, before_name, after_name, [], {})
        after_name = node.attr
        return ASTNamedAttribute(dbg_info, None, after_name, [self.visit_expr(node.value)], {})

    def visit_Name(self, node):
        dbg_info = self.get_debug_info(node)
        return ASTLoad(dbg_info, node.id)

    def visit_Constant(self, node):
        dbg_info = self.get_debug_info(node)
        if node.value is None:
            raise UnsupportedConstantLiteralException(dbg_info, "Invalid constant `None` in circuit.")
        if isinstance(node.value, int):
            return ASTConstantInteger(dbg_info, node.value)
        elif isinstance(node.value, float):
            return ASTConstantFloat(dbg_info, node.value)
        raise UnsupportedConstantLiteralException(dbg_info, f"Invalid constant value `{node.value}` in circuit")

    def visit_List(self, node):
        dbg_info = self.get_debug_info(node)
        if len(node.elts) == 0:
            raise UnsupportedLangFeatureException(dbg_info, "Cannot create an empty list (as an empty NDArray) in circuit.")
        parsed_elts = []
        for elt in node.elts:
            parsed_elts.append(self.visit_expr(elt))
        return ASTSquareBrackets(dbg_info, len(parsed_elts), parsed_elts)

    def visit_Tuple(self, node):
        dbg_info = self.get_debug_info(node)
        if len(node.elts) == 0:
            raise UnsupportedLangFeatureException(dbg_info, "Cannot create an empty tuple in circuit.")
        parsed_elts = []
        for elt in node.elts:
            parsed_elts.append(self.visit_expr(elt))
        return ASTParenthesis(dbg_info, len(parsed_elts), parsed_elts)

    def visit_Subscript(self, node):
        dbg_info = self.get_debug_info(node)
        value = self.visit_expr(node.value)
        return ASTSlicing(dbg_info, value, self.visit_slice_key(node.slice))

    def visit_Break(self, node):
        dbg_info = self.get_debug_info(node)
        return ASTBreakStatement(dbg_info)

    def visit_Continue(self, node):
        dbg_info = self.get_debug_info(node)
        return ASTContinueStatement(dbg_info)

    def visit_Return(self, node):
        dbg_info = self.get_debug_info(node)
        result = node.value
        if result is not None:
            result = self.visit_expr(result)
        return ASTReturnStatement(dbg_info, result)

    def visit_block(self, _stmts):
        stmts = []
        for stmt in _stmts:
            dbg_info = self.get_debug_info(stmt)
            if isinstance(stmt, ast.AnnAssign):
                parsed_stmt = self.visit_AnnAssign(stmt)
            elif isinstance(stmt, ast.Assign):
                parsed_stmt = self.visit_Assign(stmt)
            elif isinstance(stmt, ast.For):
                parsed_stmt = self.visit_For(stmt)
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
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                parsed_stmt = self.visit_CallStmt(stmt.value)
            elif isinstance(stmt, ast.Expr):
                raise InvalidCircuitStatementException(dbg_info, f"Statement seems to have no effect")
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
        else:
            dbg_info = self.get_debug_info(node)
            raise UnsupportedLangFeatureException(dbg_info, f"Expression transformation rule for {type(node)} is not implemented.")

    def visit_annotation(self, node, name: str, allow_circuit_anno: bool = True):
        dbg_info = self.get_debug_info(node)
        typename: str
        public: bool = False
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id in [InputAnnoName.PUBLIC, InputAnnoName.PRIVATE]:
            if not allow_circuit_anno:
                raise InvalidAnnotationException(dbg_info, f"Invalid Public/Private annotation for `{name}`, it is only allowed in circuit")
            public = node.value.id == InputAnnoName.PUBLIC
            node = node.slice
        def _inner_parser(_n: ast.Name | ast.Subscript):
            if isinstance(_n, ast.Name):
                return DTDescriptorFactory.create(dbg_info, _n.id)
            elif isinstance(_n, ast.Subscript):
                if not isinstance(_n.value, ast.Name):
                    raise InvalidAnnotationException(dbg_info, f"Invalid annotation for `{name}`")
                if isinstance(_n.slice, ast.Tuple):
                    args = []
                    for elt in _n.slice.elts:
                        if isinstance(elt, ast.Name):
                            args.append(_inner_parser(elt))
                        elif isinstance(elt, ast.Subscript):
                            args.append(_inner_parser(elt))
                        elif isinstance(elt, ast.Constant):
                            args.append(elt.value)
                        else:
                            raise InvalidAnnotationException(dbg_info,
                                                             f"Invalid annotation for `{name}`")
                    return DTDescriptorFactory.create(dbg_info, _n.value.id, tuple(args))
                elif isinstance(_n.slice, ast.Constant):
                    return DTDescriptorFactory.create(dbg_info, _n.value.id, (_n.slice.value, ))
            elif isinstance(_n, ast.Constant):
                if _n.value is None:
                    return NoneDTDescriptor()
            raise InvalidAnnotationException(dbg_info, f"Invalid annotation for `{name}`")
        return ASTAnnotation(dbg_info, _inner_parser(node), public)

    def visit_slice_key(self, node):
        dbg = self.get_debug_info(node)
        constant_none = ASTConstantNone(dbg)
        if isinstance(node, ast.Slice):
            lo, hi, step = constant_none, constant_none, constant_none
            if node.lower is not None:
                lo = self.visit_expr(node.lower)
            if node.upper is not None:
                hi = self.visit_expr(node.upper)
            if node.step is not None:
                step = self.visit_expr(node.step)
            return ASTSlice(self.get_debug_info(node), [(lo, hi, step)])
        elif isinstance(node, ast.Tuple):
            slicing_data = []
            for elt in node.elts:
                if isinstance(elt, ast.Slice):
                    lo, hi, step = constant_none, constant_none, constant_none
                    if elt.lower is not None:
                        lo = self.visit_expr(elt.lower)
                    if elt.upper is not None:
                        hi = self.visit_expr(elt.upper)
                    if elt.step is not None:
                        step = self.visit_expr(elt.step)
                    slicing_data.append((lo, hi, step))
                else:
                    slicing_data.append(self.visit_expr(elt))
            return ASTSlice(dbg, slicing_data)
        return ASTSlice(dbg, [self.visit_expr(node.slice)])


class PyZKCircuitASTTransformer(PyZKBaseASTTransformer):
    def __init__(self, source_code: str, method_name: str, chips: Dict[str, ChipObject]):
        super().__init__(source_code, method_name)
        self.chips = chips
    
    def get_debug_info(self, node) -> DebugInfo:
        return DebugInfo(self.method_name, self.source_code, True, node.lineno, node.col_offset, node.end_lineno, node.end_col_offset)

    def visit_FunctionDef(self, node):
        dbg_info = self.get_debug_info(node)
        args = self.visit_arguments(node.args)
        return ASTProgram(dbg_info, self.visit_block(node.body), args, {name: val.chip_ast for name, val in self.chips.items()})

    def visit_arguments(self, node):
        results = []
        for arg in node.args:
            dbg_info = self.get_debug_info(arg)
            name: str = arg.arg
            if arg.annotation is None:
                raise InvalidCircuitInputException(dbg_info, "Circuit input must be annotated, e.g. `x: Public[Number]`.")
            if not isinstance(arg.annotation, ast.Subscript) or not isinstance(arg.annotation.value, ast.Name):
                raise InvalidCircuitInputException(
                    dbg_info, f"Invalid input annotation for `{name}`. A valid input annotation should be like `x: Public[Number]`.")
            if arg.annotation.value.id not in [InputAnnoName.PUBLIC, InputAnnoName.PRIVATE]:
                raise InvalidCircuitInputException(
                    dbg_info, f"Invalid input annotation `{arg.annotation.value.id}` for `{name}`. It should be either `Public` or `Private`. E.g. `x: Public[Number]`.")
            public = arg.annotation.value.id == InputAnnoName.PUBLIC
            annotation = self.visit_annotation(arg.annotation, name)
            results.append(ASTProgramInput(dbg_info, public, name, annotation))
        return results


class PyZKChipASTTransformer(PyZKBaseASTTransformer):
    def __init__(self, source_code: str, method_name: str):
        super().__init__(source_code, method_name)
    
    def get_debug_info(self, node) -> DebugInfo:
        return DebugInfo(self.method_name, self.source_code, False, node.lineno, node.col_offset, node.end_lineno, node.end_col_offset)

    def visit_FunctionDef(self, node):
        dbg_info = self.get_debug_info(node)
        args = self.visit_arguments(node.args)
        if node.returns is not None:
            return_anno = self.visit_annotation(node.returns, 'Return Value', False)
        else:
            raise InvalidAnnotationException(dbg_info, "Chip must have a return annotation.")
        return ASTChip(dbg_info, self.visit_block(node.body), args, return_anno)

    def visit_arguments(self, node):
        results = []
        for arg in node.args:
            dbg_info = self.get_debug_info(arg)
            name: str = arg.arg
            if arg.annotation is None:
                raise InvalidChipInputException(dbg_info, "Chip input must be annotated, e.g. `x: Number`.")
            if not isinstance(arg.annotation, ast.Subscript) and not isinstance(arg.annotation, ast.Name):
                raise InvalidChipInputException(
                    dbg_info, f"Invalid input annotation for `{name}`. A valid input annotation should be like `x: Number`.")
            annotation = self.visit_annotation(arg.annotation, name, False)
            results.append(ASTChipInput(dbg_info, name, annotation))
        return results
