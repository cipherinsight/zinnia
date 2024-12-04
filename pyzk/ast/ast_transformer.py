import ast

from pyzk.exception.transforming import InvalidCircuitInputException, InvalidCircuitStatementException, \
    InvalidProgramException, InvalidAssignStatementException, InvalidAnnotationException, InvalidSlicingException, \
    UnsupportedOperatorException, UnsupportedConstantLiteralException, \
    UnsupportedLangFeatureException, InvalidForStatementException
from pyzk.ast.zk_ast import ASTProgramInput, ASTAnnotation, ASTProgram, ASTAssignStatement, \
    ASTLoad, ASTConstant, ASTSlicingData, ASTSlicingAssignStatement, \
    ASTForInStatement, ASTPassStatement, ASTAssertStatement, ASTCondStatement, ASTSlicing, ASTSquareBrackets, \
    ASTSlicingAssignData, ASTBreakStatement, ASTContinueStatement, ASTBinaryOperator, ASTUnaryOperator, \
    ASTNamedAttribute, \
    ASTExprAttribute, ASTParenthesis
from pyzk.util.datatype_name import DataTypeName
from pyzk.util.input_anno_name import InputAnnoName
from pyzk.opdef.operator_factory import Operators
from pyzk.util.source_pos_info import SourcePosInfo


def _get_source_pos_info(node) -> SourcePosInfo:
    return SourcePosInfo(node.lineno, node.col_offset, node.end_lineno, node.end_col_offset)


class PyZKASTTransformer(ast.NodeTransformer):
    def visit(self, node):
        if isinstance(node, ast.FunctionDef):
            return self.visit_FunctionDef(node)
        raise InvalidProgramException(None, "Invalid program passed to the compiler! The program must be a function.")

    def visit_FunctionDef(self, node):
        source_pos_info = _get_source_pos_info(node)
        args = self.visit_arguments(node.args)
        return ASTProgram(source_pos_info, self.visit_block(node.body), args)

    def visit_arguments(self, node):
        results = []
        for arg in node.args:
            source_pos_info = _get_source_pos_info(arg)
            name: str = arg.arg
            if arg.annotation is None:
                raise InvalidCircuitInputException(source_pos_info, "Circuit input must be annotated, e.g. `x: Public[Number]`.")
            if not isinstance(arg.annotation, ast.Subscript) or not isinstance(arg.annotation.value, ast.Name):
                raise InvalidCircuitInputException(
                    source_pos_info, f"Invalid input annotation for `{name}`. A valid input annotation should be like `x: Public[Number]`.")
            if arg.annotation.value.id not in [InputAnnoName.PUBLIC, InputAnnoName.PRIVATE]:
                raise InvalidCircuitInputException(
                    source_pos_info, f"Invalid input annotation `{arg.annotation.value.id}` for `{name}`. It should be either `Public` or `Private`. E.g. `x: Public[Number]`.")
            public = arg.annotation.value.id == InputAnnoName.PUBLIC
            annotation = self.visit_annotation(arg.annotation, name)
            results.append(ASTProgramInput(source_pos_info, public, name, annotation))
        return results

    def visit_For(self, node):
        if not isinstance(node.target, ast.Name):
            target_pos_info = _get_source_pos_info(node.target)
            raise InvalidForStatementException(target_pos_info, 'In for statement, the variable before keyword "in" must be a name.')
        target_name = node.target.id
        iter_expr = self.visit_expr(node.iter)
        return ASTForInStatement(_get_source_pos_info(node.iter), target_name, iter_expr, self.visit_block(node.body))

    def visit_Assert(self, node):
        source_pos_info = _get_source_pos_info(node)
        test = self.visit_expr(node.test)
        return ASTAssertStatement(source_pos_info, test)

    def visit_Pass(self, node):
        source_pos_info = _get_source_pos_info(node)
        return ASTPassStatement(source_pos_info)

    def visit_If(self, node):
        source_pos_info = _get_source_pos_info(node)
        test = self.visit_expr(node.test)
        return ASTCondStatement(source_pos_info, test, self.visit_block(node.body), self.visit_block(node.orelse))

    def visit_Assign(self, node):
        source_pos_info = _get_source_pos_info(node)
        if len(node.targets) != 1:
            raise InvalidAssignStatementException(source_pos_info, "The assignment statement does not support multiple assignments (unpacking).")
        if isinstance(node.targets[0], ast.Name):
            identifier_name = node.targets[0].id
            expr = self.visit_expr(node.value)
            return ASTAssignStatement(source_pos_info, identifier_name, expr, None)
        elif isinstance(node.targets[0], ast.Subscript):
            name, slicing_datas = self.visit_slicing_assignee(node.targets[0])
            expr = self.visit_expr(node.value)
            return ASTSlicingAssignStatement(source_pos_info, name, slicing_datas, expr, None)
        raise InvalidAssignStatementException(source_pos_info, "The value to be assigned must be an identifier name or name with slicing.")

    def visit_AugAssign(self, node):
        source_pos_info = _get_source_pos_info(node)
        if isinstance(node.op, ast.Add):
            op_cls = Operators.NoCls.ADD
        elif isinstance(node.op, ast.Sub):
            op_cls = Operators.NoCls.SUB
        elif isinstance(node.op, ast.Div):
            op_cls = Operators.NoCls.DIV
        elif isinstance(node.op, ast.Mult):
            op_cls = Operators.NoCls.MUL
        elif isinstance(node.op, ast.MatMult):
            op_cls = Operators.NoCls.MAT_MUL
        else:
            raise InvalidAssignStatementException(source_pos_info, f"Invalid augmented assignment operator {type(node.op)}")
        if isinstance(node.target, ast.Name):
            identifier_name = node.target.id
            expr = self.visit_expr(node.value)
            return ASTAssignStatement(
                source_pos_info, identifier_name, ASTBinaryOperator(
                    source_pos_info, op_cls, ASTLoad(_get_source_pos_info(node.target), identifier_name), expr
                ), None)
        elif isinstance(node.target, ast.Subscript):
            name, slicing_datas = self.visit_slicing_assignee(node.target)
            origin_val = self.visit_expr(node.target)
            expr = self.visit_expr(node.value)
            return ASTSlicingAssignStatement(
                source_pos_info, name, slicing_datas, ASTBinaryOperator(
                    _get_source_pos_info(node.value), op_cls, origin_val, expr
                ), None)
        raise InvalidAssignStatementException(source_pos_info, "The value to be assigned must be an identifier name or name with slicing.")

    def visit_AnnAssign(self, node):
        source_pos_info = _get_source_pos_info(node)
        if not isinstance(node.target, ast.Name):
            raise InvalidAssignStatementException(source_pos_info, "The value to be assigned must be an identifier name.")
        if isinstance(node.target, ast.Name):
            identifier_name = node.target.id
            expr = self.visit_expr(node.value)
            annotation = self.visit_annotation(node.annotation, identifier_name)
            return ASTAssignStatement(source_pos_info, identifier_name, expr, annotation)
        elif isinstance(node.target, ast.Subscript):
            name, slicing_datas = self.visit_slicing_assignee(node.target)
            expr = self.visit_expr(node.value)
            annotation = self.visit_annotation(node.annotation, name)
            return ASTSlicingAssignStatement(source_pos_info, name, slicing_datas, expr, annotation)
        raise InvalidAssignStatementException(source_pos_info, "The value to be assigned must be an identifier name or name with slicing.")

    def visit_BinOp(self, node):
        source_pos_info = _get_source_pos_info(node)
        left = self.visit_expr(node.left)
        right = self.visit_expr(node.right)
        if isinstance(node.op, ast.Add):
            return ASTBinaryOperator(source_pos_info, Operators.NoCls.ADD, left, right)
        elif isinstance(node.op, ast.Mult):
            return ASTBinaryOperator(source_pos_info, Operators.NoCls.MUL, left, right)
        elif isinstance(node.op, ast.MatMult):
            return ASTBinaryOperator(source_pos_info, Operators.NoCls.MAT_MUL, left, right)
        elif isinstance(node.op, ast.Div):
            return ASTBinaryOperator(source_pos_info, Operators.NoCls.DIV, left, right)
        elif isinstance(node.op, ast.Sub):
            return ASTBinaryOperator(source_pos_info, Operators.NoCls.SUB, left, right)
        else:
            raise UnsupportedOperatorException(source_pos_info, f"Invalid binary operator {type(node.op).__name__} in circuit.")

    def visit_Compare(self, node):
        source_pos_info = _get_source_pos_info(node)
        left = self.visit_expr(node.left)
        comparators = []
        for com in node.comparators:
            comparators.append(self.visit_expr(com))
        assert len(node.ops) == len(comparators)
        comparators = [left] + comparators
        compare_expr_list = []
        for i, op in enumerate(node.ops):
            if isinstance(op, ast.GtE):
                compare_expr_list.append(ASTBinaryOperator(source_pos_info, Operators.NoCls.GTE, comparators[i], comparators[i + 1]))
            elif isinstance(op, ast.LtE):
                compare_expr_list.append(ASTBinaryOperator(source_pos_info, Operators.NoCls.LTE, comparators[i], comparators[i + 1]))
            elif isinstance(op, ast.Gt):
                compare_expr_list.append(ASTBinaryOperator(source_pos_info, Operators.NoCls.GT, comparators[i], comparators[i + 1]))
            elif isinstance(op, ast.Lt):
                compare_expr_list.append(ASTBinaryOperator(source_pos_info, Operators.NoCls.LT, comparators[i], comparators[i + 1]))
            elif isinstance(op, ast.Eq):
                compare_expr_list.append(ASTBinaryOperator(source_pos_info, Operators.NoCls.EQ, comparators[i], comparators[i + 1]))
            elif isinstance(op, ast.NotEq):
                compare_expr_list.append(ASTBinaryOperator(source_pos_info, Operators.NoCls.NE, comparators[i], comparators[i + 1]))
            else:
                raise UnsupportedOperatorException(source_pos_info, f"Invalid compare operator {type(op).__name__} in circuit.")
        finalized = compare_expr_list[0]
        for expr in compare_expr_list[1:]:
            finalized = ASTBinaryOperator(source_pos_info, Operators.NoCls.AND, finalized, expr)
        return finalized

    def visit_BoolOp(self, node):
        source_pos_info = _get_source_pos_info(node)
        values = []
        for val in node.values:
            values.append(self.visit_expr(val))
        assert len(values) > 1
        if isinstance(node.op, ast.And):
            base_node = values[0]
            for val in values[1:]:
                base_node = ASTBinaryOperator(source_pos_info, Operators.NoCls.AND, base_node, val)
            return base_node
        elif isinstance(node.op, ast.Or):
            base_node = values[0]
            for val in values[1:]:
                base_node = ASTBinaryOperator(source_pos_info, Operators.NoCls.OR, base_node, val)
            return base_node
        else:
            raise UnsupportedOperatorException(source_pos_info, f"Invalid boolean operator {type(node.op).__name__} in circuit.")

    def visit_UnaryOp(self, node):
        source_pos_info = _get_source_pos_info(node)
        value = self.visit_expr(node.operand)
        if isinstance(node.op, ast.Not):
            return ASTUnaryOperator(source_pos_info, Operators.NoCls.NOT, value)
        elif isinstance(node.op, ast.USub):
            return ASTUnaryOperator(source_pos_info, Operators.NoCls.USUB, value)
        elif isinstance(node.op, ast.UAdd):
            return value
        else:
            raise UnsupportedOperatorException(source_pos_info, f"Invalid unary operator {type(node.op)} in circuit.")

    def visit_Call(self, node):
        source_pos_info = _get_source_pos_info(node)
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                before_name = node.func.value.id
                after_name = node.func.attr
                return ASTNamedAttribute(
                    source_pos_info, before_name, after_name,
                    [self.visit_expr(arg) for arg in node.args],
                    {kwarg.arg: self.visit_expr(kwarg.value) for kwarg in node.keywords}
                )
            else:
                after_name = node.func.attr
                return ASTExprAttribute(
                    source_pos_info, self.visit_expr(node.func.value), after_name,
                    [self.visit_expr(arg) for arg in node.args],
                    {kwarg.arg: self.visit_expr(kwarg.value) for kwarg in node.keywords}
                )
        elif isinstance(node.func, ast.Name):
            return ASTNamedAttribute(
                source_pos_info, None, node.func.id,
                [self.visit_expr(arg) for arg in node.args],
                {kwarg.arg: self.visit_expr(kwarg.value) for kwarg in node.keywords}
            )
        raise UnsupportedOperatorException(source_pos_info, f"Invalid call function {type(node.func)}. Only a static specified function name is supported here.")

    def visit_Attribute(self, node):
        source_pos_info = _get_source_pos_info(node)
        if isinstance(node.value, ast.Name):
            after_name = node.attr
            before_name = node.value.id
            return ASTNamedAttribute(source_pos_info, before_name, after_name, [], {})
        after_name = node.func.attr
        return ASTNamedAttribute(source_pos_info, None, after_name, [self.visit_expr(node.func.value)], {})

    def visit_Name(self, node):
        source_pos_info = _get_source_pos_info(node)
        return ASTLoad(source_pos_info, node.id)

    def visit_Constant(self, node):
        source_pos_info = _get_source_pos_info(node)
        if node.value is None:
            raise UnsupportedConstantLiteralException(source_pos_info, "Invalid constant `None` in circuit.")
        elif isinstance(node.value, float):
            raise UnsupportedConstantLiteralException(source_pos_info, f"Invalid float constant `{node.value}` in circuit. ZK circuits only accept scalar numbers.")
        return ASTConstant(source_pos_info, node.value)

    def visit_List(self, node):
        source_pos_info = _get_source_pos_info(node)
        if len(node.elts) == 0:
            raise UnsupportedLangFeatureException(source_pos_info, "Cannot create an empty list (as an empty NDArray) in circuit.")
        parsed_elts = []
        for elt in node.elts:
            parsed_elts.append(self.visit_expr(elt))
        return ASTSquareBrackets(source_pos_info, len(parsed_elts), parsed_elts)

    def visit_Tuple(self, node):
        source_pos_info = _get_source_pos_info(node)
        if len(node.elts) == 0:
            raise UnsupportedLangFeatureException(source_pos_info, "Cannot create an empty tuple in circuit.")
        parsed_elts = []
        for elt in node.elts:
            parsed_elts.append(self.visit_expr(elt))
        return ASTParenthesis(source_pos_info, len(parsed_elts), parsed_elts)

    def visit_Subscript(self, node):
        source_pos_info = _get_source_pos_info(node)
        value = self.visit_expr(node.value)
        if isinstance(node.slice, ast.Slice):
            lo, hi, step = None, None, None
            if node.slice.lower is not None:
                lo = self.visit_expr(node.slice.lower)
            if node.slice.upper is not None:
                hi = self.visit_expr(node.slice.upper)
            if node.slice.step is not None:
                step = self.visit_expr(node.slice.step)
            return ASTSlicing(source_pos_info, value, ASTSlicingData(source_pos_info, [(lo, hi, step)]))
        elif isinstance(node.slice, ast.Tuple):
            slicing_data = []
            for elt in node.slice.elts:
                if isinstance(elt, ast.Slice):
                    lo, hi, step = None, None, None
                    if elt.lower is not None:
                        lo = self.visit_expr(elt.lower)
                    if elt.upper is not None:
                        hi = self.visit_expr(elt.upper)
                    if elt.step is not None:
                        step = self.visit_expr(elt.step)
                    slicing_data.append((lo, hi, step))
                else:
                    slicing_data.append(self.visit_expr(elt))
            return ASTSlicing(source_pos_info, value, ASTSlicingData(source_pos_info, slicing_data))
        else:
            return ASTSlicing(source_pos_info, value, ASTSlicingData(source_pos_info, [self.visit_expr(node.slice)]))

    def visit_Break(self, node):
        source_pos_info = _get_source_pos_info(node)
        return ASTBreakStatement(source_pos_info)

    def visit_Continue(self, node):
        source_pos_info = _get_source_pos_info(node)
        return ASTContinueStatement(source_pos_info)

    def visit_block(self, _stmts):
        stmts = []
        for stmt in _stmts:
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
            else:
                source_pos_info = _get_source_pos_info(stmt)
                raise InvalidCircuitStatementException(source_pos_info, f"Invalid circuit statement defined: {type(stmt)}.")
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
            source_pos_info = _get_source_pos_info(node)
            raise UnsupportedLangFeatureException(source_pos_info, f"Expression transformation rule for {type(node)} is not implemented.")

    def visit_annotation(self, node, name: str):
        source_pos_info = _get_source_pos_info(node)
        typename: str
        public: bool = False
        size = tuple()
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id in [InputAnnoName.PUBLIC, InputAnnoName.PRIVATE]:
            public = node.value.id == InputAnnoName.PUBLIC
            node = node.slice
        if isinstance(node, ast.Name):
            if node.id != DataTypeName.NUMBER:
                raise InvalidAnnotationException(
                    source_pos_info, f"Invalid annotation `{node.id}` for `{name}`. If you are going to annotate a NDArray, please specify the size, e.g. `NDArray[10, 10]` or `NDArray[16]`.")
            else:
                typename = DataTypeName.NUMBER
        elif isinstance(node, ast.Subscript):
            if not isinstance(node.value, ast.Name):
                raise InvalidAnnotationException(
                    source_pos_info, f"Invalid circuit input annotation for `{name}`. A valid input annotation should be like `x: NDArray[10, 10]` or `x: NDArray[16]`.")
            if node.value.id == DataTypeName.NDARRAY:
                typename = DataTypeName.NDARRAY
                if isinstance(node.slice, ast.Tuple):
                    if not all([isinstance(elt, ast.Constant) for elt in node.slice.elts]):
                        raise InvalidAnnotationException(
                            source_pos_info, f"Invalid circuit input annotation on `NDArray` for `{name}`. A valid input annotation should be like `x: NDArray[10, 10]`. All dimension sizes should be constants.")
                    size = tuple([elt.value for elt in node.slice.elts])
                elif isinstance(node.slice, ast.Constant):
                    size = tuple([node.slice.value])
                else:
                    raise InvalidAnnotationException(
                        source_pos_info, f"Invalid circuit input annotation on `NDArray` for `{name}`. A valid input annotation should be like `x: NDArray[10, 10]`.")
            else:
                raise InvalidAnnotationException(
                    source_pos_info, f"Invalid annotation on `{node.value.id}` for `{name}`. It should be either `Number` or `NDArray`.")
        else:
            raise InvalidAnnotationException(source_pos_info, f'Unsupported annotation type for `{name}`.')
        return ASTAnnotation(source_pos_info, typename, size, public)

    def visit_slicing_assignee(self, node):
        source_pos_info = _get_source_pos_info(node)
        assert isinstance(node, ast.Subscript)
        def _visit_slicing_assignee_helper(subscript_node):
            _assignee_name = None
            ans = []
            if isinstance(subscript_node.value, ast.Name):
                _assignee_name = subscript_node.value.id
            elif isinstance(subscript_node.value, ast.Subscript):
                _assignee_name, ans = _visit_slicing_assignee_helper(subscript_node.value)
            else:
                raise InvalidSlicingException(source_pos_info, "This slicing assign format is not supported.")
            if isinstance(subscript_node.slice, ast.Slice):
                lo, hi, step = None, None, None
                if subscript_node.slice.lower is not None:
                    lo = self.visit_expr(subscript_node.slice.lower)
                if subscript_node.slice.upper is not None:
                    hi = self.visit_expr(subscript_node.slice.upper)
                if subscript_node.slice.step is not None:
                    step = self.visit_expr(subscript_node.slice.step)
                return _assignee_name, ans + [ASTSlicingData(source_pos_info, [(lo, hi, step)])]
            elif isinstance(subscript_node.slice, ast.Tuple):
                _slicing_data = []
                for elt in subscript_node.slice.elts:
                    if isinstance(elt, ast.Slice):
                        lo, hi, step = None, None, None
                        if elt.lower is not None:
                            lo = self.visit_expr(elt.lower)
                        if elt.upper is not None:
                            hi = self.visit_expr(elt.upper)
                        if elt.step is not None:
                            step = self.visit_expr(elt.step)
                        _slicing_data.append((lo, hi, step))
                    else:
                        _slicing_data.append(self.visit_expr(elt))
                return _assignee_name, ans + [ASTSlicingData(source_pos_info, _slicing_data)]
            return _assignee_name, ans + [ASTSlicingData(source_pos_info, [self.visit_expr(subscript_node.slice)])]

        name, slicing_data_list = _visit_slicing_assignee_helper(node)
        return name, ASTSlicingAssignData(source_pos_info, slicing_data_list)
