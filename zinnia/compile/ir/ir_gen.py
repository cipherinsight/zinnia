from typing import List, Tuple, Dict

from zinnia.config.zinnia_config import ZinniaConfig
from zinnia.compile.ast.ast_formatted_value import ASTFormattedValue
from zinnia.compile.ast.ast_joined_str import ASTJoinedStr
from zinnia.compile.ast.ast_starred import ASTStarredExpr
from zinnia.compile.ir.ir_graph import IRGraph
from zinnia.compile.builder.builder_impl import IRBuilderImpl
from zinnia.compile.triplet import Value, StringValue, ListValue, TupleValue
from zinnia.compile.ast import ASTComponent, ASTCircuit, ASTAssignStatement, ASTPassStatement, \
    ASTExpression, ASTForInStatement, ASTCondStatement, ASTConstantFloat, ASTConstantInteger, \
    ASTSubscriptExp, ASTLoad, ASTAssertStatement, ASTSquareBrackets, ASTBreakStatement, ASTContinueStatement, \
    ASTBinaryOperator, ASTNamedAttribute, ASTExprAttribute, ASTParenthesis, ASTChip, ASTReturnStatement, \
    ASTUnaryOperator, ASTConstantNone, ASTNameAssignTarget, ASTSubscriptAssignTarget, \
    ASTListAssignTarget, ASTTupleAssignTarget, ASTAssignTarget, ASTExprStatement, ASTConstantString, ASTGeneratorExp, \
    ASTGenerator, ASTCondExp, ASTWhileStatement
from zinnia.compile.ir.ir_ctx import IRContext
from zinnia.debug.exception import VariableNotFoundError, NotInLoopError, \
    InterScopeError, UnsupportedLangFeatureException, UnreachableStatementError, OperatorOrChipNotFoundException, \
    ControlEndWithoutReturnError, ReturnDatatypeMismatchError, \
    ChipArgumentsError, UnpackingError, StaticInferenceError, LoopLimitExceedError, RecursionLimitExceedError

from zinnia.internal.internal_chip_object import InternalChipObject
from zinnia.internal.internal_external_func_object import InternalExternalFuncObject
from zinnia.op_def.operator_factory import Operators
from zinnia.compile.type_sys import DTDescriptorFactory, NoneDTDescriptor, FloatDTDescriptor, IntegerDTDescriptor


class IRGenerator:
    def __init__(self, config: ZinniaConfig):
        self._ir_builder = IRBuilderImpl()
        self._ir_ctx = IRContext(self._ir_builder)
        self._registered_chips = {}
        self._registered_externals = {}
        self._next_external_call_id = 1
        self._config = config

    def generate(
            self,
            ast_root: ASTCircuit,
            chips: Dict[str, InternalChipObject],
            externals: Dict[str, InternalExternalFuncObject]
    ) -> IRGraph:
        self._registered_chips = chips
        self._registered_externals = externals
        self.visit(ast_root)
        return self._ir_builder.export_ir_graph()

    def visit(self, component: ASTComponent):
        typename = type(component).__name__
        method_name = 'visit_' + typename
        method = getattr(self, method_name, None)
        if method is None:
            raise NotImplementedError(method_name)
        return method(component)

    def visit_ASTCircuit(self, n: ASTCircuit):
        for i, inp in enumerate(n.inputs):
            ptr = self._ir_builder.op_input((0, i), inp.annotation.dt, inp.annotation.kind, dbg=n.dbg)
            self._ir_ctx.set(inp.name, ptr)
        self._register_global_datatypes()
        for i, stmt in enumerate(n.block):
            self.visit(stmt)
        return None

    def visit_ASTPassStatement(self, n: ASTPassStatement):
        return None

    def visit_ASTAssignStatement(self, n: ASTAssignStatement):
        if self._ir_ctx.check_return_guaranteed() or self._ir_ctx.check_loop_terminated_guaranteed():
            return None
        val_ptr = self.visit(n.value)
        for target in n.targets:
            self._do_recursive_assign(target, val_ptr, True)
        return None

    def visit_ASTForInStatement(self, n: ASTForInStatement):
        if self._ir_ctx.check_return_guaranteed() or self._ir_ctx.check_loop_terminated_guaranteed():
            return None
        iter_elts = self._ir_builder.op_iter(self.visit(n.iter_expr))
        loop_body_return_guaranteed = False
        loop_terminated = False
        self._ir_ctx.loop_enter()
        for loop_index_ptr in iter_elts.values():
            self._do_recursive_assign(n.target, loop_index_ptr, False)
            self._ir_ctx.loop_reiter()
            for _, stmt in enumerate(n.block):
                break_condition = self._ir_ctx.get_break_condition_value()
                if break_condition.val() is not None and break_condition.val() == 0:
                    loop_terminated = True
                    break
                if self._ir_ctx.check_return_guaranteed():
                    loop_body_return_guaranteed = True
                    loop_terminated = True
                    break
                if self._ir_ctx.check_loop_terminated_guaranteed():
                    loop_terminated = True
                    break
                self.visit(stmt)
            if loop_terminated:
                break
        break_condition = self._ir_ctx.get_break_condition_value()
        self._ir_ctx.loop_leave()
        if break_condition.val() is None or break_condition.val() != 0:
            self._ir_ctx.if_enter(break_condition)
            for _, stmt in enumerate(n.orelse):
                self.visit(stmt)
            orelse_scope = self._ir_ctx.if_leave()
            if break_condition.val() is not None and break_condition.val() != 0 and orelse_scope.is_return_guaranteed():
                self._ir_ctx.set_return_guarantee()
        if loop_body_return_guaranteed:
            self._ir_ctx.set_return_guarantee()
        return None

    def visit_ASTWhileStatement(self, n: ASTWhileStatement):
        if self._ir_ctx.check_return_guaranteed() or self._ir_ctx.check_loop_terminated_guaranteed():
            return None
        loop_body_has_return = False
        loop_terminated = False
        loop_quota = self._config.loop_limit() + 1
        self._ir_ctx.loop_enter()
        while True:
            self._ir_ctx.loop_reiter()
            test_expr = self._ir_builder.op_bool_cast(self.visit(n.test_expr), dbg=n.dbg)
            loop_quota -= 1
            if loop_quota <= 0:
                if test_expr.val() is None:
                    # TODO: raise a warning here
                    self._ir_builder.op_assert(self._ir_builder.ir_constant_int(0), test_expr, dbg=n.dbg)
                    break
                else:
                    raise LoopLimitExceedError(n.dbg, "Loop limit exceeded on while. Please check for infinite loops, or increase the loop limit.")
            elif test_expr.val() == 0:
                break
            if test_expr.val() is None:
                self._ir_ctx.loop_break(self._ir_builder.ir_logical_not(test_expr))
            for _, stmt in enumerate(n.block):
                self.visit(stmt)
                break_condition = self._ir_ctx.get_break_condition_value()
                if break_condition.val() is not None and break_condition.val() == 0:
                    loop_terminated = True
                    break
                if self._ir_ctx.check_return_guaranteed():
                    loop_body_has_return = True
                    loop_terminated = True
                    break
                if self._ir_ctx.check_loop_terminated_guaranteed():
                    loop_terminated = True
                    break
            if loop_terminated:
                break
        break_condition = self._ir_ctx.get_break_condition_value()
        self._ir_ctx.loop_leave()
        if break_condition.val() is None or break_condition.val() != 0:
            self._ir_ctx.if_enter(break_condition)
            for _, stmt in enumerate(n.orelse):
                self.visit(stmt)
            orelse_scope = self._ir_ctx.if_leave()
            if break_condition.val() is not None and break_condition.val() != 0 and orelse_scope.is_return_guaranteed():
                self._ir_ctx.set_return_guarantee()
        if loop_body_has_return:
            self._ir_ctx.set_return_guarantee()
        return None

    def visit_ASTBreakStatement(self, n: ASTBreakStatement):
        if self._ir_ctx.check_return_guaranteed() or self._ir_ctx.check_loop_terminated_guaranteed():
            return None
        if not self._ir_ctx.is_in_loop():
            raise NotInLoopError(n.dbg, "Invalid break statement here outside the loop.")
        self._ir_ctx.loop_break()
        self._ir_ctx.set_terminated_guarantee()
        return None

    def visit_ASTContinueStatement(self, n: ASTContinueStatement):
        if self._ir_ctx.check_return_guaranteed() or self._ir_ctx.check_loop_terminated_guaranteed():
            return None
        if not self._ir_ctx.is_in_loop():
            raise NotInLoopError(n.dbg, "Invalid continue statement here outside the loop.")
        self._ir_ctx.loop_continue()
        return None

    def visit_ASTCondStatement(self, n: ASTCondStatement):
        if self._ir_ctx.check_return_guaranteed() or self._ir_ctx.check_loop_terminated_guaranteed():
            return None
        cond_ptr = self.visit(n.cond)
        true_cond_ptr = self._ir_builder.op_bool_cast(cond_ptr, dbg=n.dbg)
        false_cond_ptr = self._ir_builder.ir_logical_not(true_cond_ptr, dbg=n.dbg)
        scope_true = None
        if true_cond_ptr.val() is None or true_cond_ptr.val() != 0:
            self._ir_ctx.if_enter(true_cond_ptr)
            for _, stmt in enumerate(n.t_block):
                self.visit(stmt)
            scope_true = self._ir_ctx.if_leave()
        scope_false = None
        if false_cond_ptr.val() is None or false_cond_ptr.val() != 0:
            self._ir_ctx.if_enter(false_cond_ptr)
            for _, stmt in enumerate(n.f_block):
                self.visit(stmt)
            scope_false = self._ir_ctx.if_leave()
        # update return guarantee
        if scope_true is not None and true_cond_ptr.val() is not None and true_cond_ptr.val() != 0 and scope_true.is_return_guaranteed():
            self._ir_ctx.set_return_guarantee()
        elif scope_false is not None and false_cond_ptr.val() is not None and false_cond_ptr.val() != 0 and scope_false.is_return_guaranteed():
            self._ir_ctx.set_return_guarantee()
        elif scope_true is not None and scope_false is not None and scope_true.is_return_guaranteed() and scope_false.is_return_guaranteed():
            self._ir_ctx.set_return_guarantee()
        # update terminated guarantee
        if scope_true is not None and true_cond_ptr.val() is not None and true_cond_ptr.val() != 0:
            if scope_true.is_terminated_guaranteed() or scope_true.is_return_guaranteed():
                if self._ir_ctx.is_in_loop():
                    self._ir_ctx.set_terminated_guarantee()
        elif scope_false is not None and false_cond_ptr.val() is not None and false_cond_ptr.val() != 0 and scope_false.is_terminated_guaranteed():
            if scope_false.is_terminated_guaranteed() or scope_false.is_return_guaranteed():
                if self._ir_ctx.is_in_loop():
                    self._ir_ctx.set_terminated_guarantee()
        elif scope_true is not None and scope_false is not None:
            if (scope_false.is_terminated_guaranteed() or scope_false.is_return_guaranteed()) and (scope_true.is_terminated_guaranteed() or scope_true.is_return_guaranteed()):
                if self._ir_ctx.is_in_loop():
                    self._ir_ctx.set_terminated_guarantee()
        return None

    def visit_ASTAssertStatement(self, n: ASTAssertStatement):
        if self._ir_ctx.check_return_guaranteed() or self._ir_ctx.check_loop_terminated_guaranteed():
            return None
        test = self.visit(n.expr)
        return self._ir_builder.op_assert(test, self._ir_ctx.get_condition_value_for_assertion(), dbg=n.dbg)

    def visit_ASTReturnStatement(self, n: ASTReturnStatement):
        if self._ir_ctx.check_return_guaranteed() or self._ir_ctx.check_loop_terminated_guaranteed():
            return None
        if not self._ir_ctx.is_in_chip():
            raise UnsupportedLangFeatureException(n.dbg, "Return statements in circuits is not supported, it is only supported in chips")
        if n.expr is not None:
            val = self.visit(n.expr)
            return_dt = val.type()
        else:
            val = self._ir_builder.op_constant_none()
            return_dt = NoneDTDescriptor()
        expected_return_dt = self._ir_ctx.get_return_dtype()
        if return_dt != expected_return_dt:
            raise ReturnDatatypeMismatchError(n.dbg, "Return datatype mismatch annotated return datatype")
        self._ir_ctx.register_return(val)
        self._ir_ctx.set_return_guarantee()
        return None

    def visit_ASTExprStatement(self, n: ASTExprStatement):
        if self._ir_ctx.check_return_guaranteed() or self._ir_ctx.check_loop_terminated_guaranteed():
            return None
        return self.visit(n.expr)

    def visit_ASTBinaryOperator(self, n: ASTBinaryOperator):
        lhs_expr = self.visit(n.lhs)
        rhs_expr = self.visit(n.rhs)
        if n.operator == n.Op.ADD:
            return self._ir_builder.op_add(lhs_expr, rhs_expr, dbg=n.dbg)
        elif n.operator == n.Op.SUB:
            return self._ir_builder.op_subtract(lhs_expr, rhs_expr, dbg=n.dbg)
        elif n.operator == n.Op.MUL:
            return self._ir_builder.op_multiply(lhs_expr, rhs_expr, dbg=n.dbg)
        elif n.operator == n.Op.DIV:
            return self._ir_builder.op_divide(lhs_expr, rhs_expr, dbg=n.dbg)
        elif n.operator == n.Op.MOD:
            return self._ir_builder.op_modulo(lhs_expr, rhs_expr, dbg=n.dbg)
        elif n.operator == n.Op.FLOOR_DIV:
            return self._ir_builder.op_floor_divide(lhs_expr, rhs_expr, dbg=n.dbg)
        elif n.operator == n.Op.MAT_MUL:
            return self._ir_builder.op_mat_mul(lhs_expr, rhs_expr, dbg=n.dbg)
        elif n.operator == n.Op.EQ:
            return self._ir_builder.op_equal(lhs_expr, rhs_expr, dbg=n.dbg)
        elif n.operator == n.Op.NE:
            return self._ir_builder.op_not_equal(lhs_expr, rhs_expr, dbg=n.dbg)
        elif n.operator == n.Op.LT:
            return self._ir_builder.op_less_than(lhs_expr, rhs_expr, dbg=n.dbg)
        elif n.operator == n.Op.LTE:
            return self._ir_builder.op_less_than_or_equal(lhs_expr, rhs_expr, dbg=n.dbg)
        elif n.operator == n.Op.GT:
            return self._ir_builder.op_greater_than(lhs_expr, rhs_expr, dbg=n.dbg)
        elif n.operator == n.Op.GTE:
            return self._ir_builder.op_greater_than_or_equal(lhs_expr, rhs_expr, dbg=n.dbg)
        elif n.operator == n.Op.AND:
            return self._ir_builder.ir_logical_and(lhs_expr, rhs_expr, dbg=n.dbg)
        elif n.operator == n.Op.OR:
            return self._ir_builder.ir_logical_or(lhs_expr, rhs_expr, dbg=n.dbg)
        elif n.operator == n.Op.POW:
            return self._ir_builder.op_power(lhs_expr, rhs_expr, dbg=n.dbg)
        raise NotImplementedError(f"Internal Error: Binary Operator {n.operator} not implemented")

    def visit_ASTUnaryOperator(self, n: ASTUnaryOperator):
        operand_expr = self.visit(n.operand)
        if n.operator == n.Op.NOT:
            return self._ir_builder.op_unary_not(operand_expr, dbg=n.dbg)
        if n.operator == n.Op.USUB:
            return self._ir_builder.op_unary_sub(operand_expr, dbg=n.dbg)
        if n.operator == n.Op.UADD:
            return self._ir_builder.op_unary_add(operand_expr, dbg=n.dbg)
        raise NotImplementedError(f"Internal Error: Unary Operator {n.operator} not implemented")

    def visit_ASTNamedAttribute(self, n: ASTNamedAttribute):
        self_arg = []
        visited_args = []
        for arg in n.args:
            if isinstance(arg, ASTStarredExpr):
                iter_values = self._ir_builder.op_iter(self.visit(arg.value), arg.value.dbg)
                visited_args.extend(iter_values.values())
            else:
                visited_args.append(self.visit(arg))
        visited_kwargs = {k: self.visit(arg) for k, arg in n.kwargs.items()}
        if n.target is None or DTDescriptorFactory.is_typename(n.target):
            chip = self._registered_chips.get(n.member, None)
            if n.target is None and chip is not None:
                return self.visit_chip_call(chip, visited_args, visited_kwargs)
            external_func = self._registered_externals.get(n.member, None)
            if external_func is not None:
                return self.visit_external_call(external_func, visited_args, visited_kwargs)
            operator = Operators.instantiate_operator(n.member, n.target)
            if operator is None:
                raise OperatorOrChipNotFoundException(n.dbg, f"Operator or Chip `{n.member}` not found")
        else:
            if self._ir_ctx.exists(n.target):
                target_ptr = self._ir_ctx.get(n.target)
                dt = target_ptr.type()
                operator = Operators.instantiate_operator(n.member, dt.get_typename())
                self_arg = [target_ptr]
            elif n.target in Operators.get_namespaces():
                operator = Operators.instantiate_operator(n.member, n.target)
            else:
                raise VariableNotFoundError(n.dbg, f'Variable {n.target} referenced but not defined.')
            if operator is None:
                raise OperatorOrChipNotFoundException(n.dbg, f"Operator `{n.target}.{n.member}` not found")
        return self._ir_builder.create_op(operator, self._ir_ctx.get_condition_value(), self_arg + visited_args, visited_kwargs, dbg=n.dbg)

    def visit_ASTExprAttribute(self, n: ASTExprAttribute):
        target_ptr = self.visit(n.target)
        dt = target_ptr.type()
        operator = Operators.instantiate_operator(n.member, dt.get_typename())
        if operator is None:
            raise OperatorOrChipNotFoundException(n.dbg, f"Operator or Chip `{n.member}` not found")
        visited_args = []
        for arg in n.args:
            if isinstance(arg, ASTStarredExpr):
                iter_values = self._ir_builder.op_iter(self.visit(arg.value), arg.value.dbg)
                visited_args.extend(iter_values.values())
            else:
                visited_args.append(self.visit(arg))
        return self._ir_builder.create_op(
            operator,
            self._ir_ctx.get_condition_value(),
            [target_ptr] + visited_args,
            {k: self.visit(arg) for k, arg in n.kwargs.items()},
            dbg=n.dbg
        )

    def visit_ASTConstantFloat(self, n: ASTConstantFloat):
        return self._ir_builder.ir_constant_float(n.value, dbg=n.dbg)

    def visit_ASTConstantInteger(self, n: ASTConstantInteger):
        return self._ir_builder.ir_constant_int(n.value, dbg=n.dbg)

    def visit_ASTConstantNone(self, n: ASTConstantNone):
        return self._ir_builder.op_constant_none(dbg=n.dbg)

    def visit_ASTConstantString(self, n: ASTConstantString):
        return self._ir_builder.ir_constant_str(n.value, dbg=n.dbg)

    def visit_ASTSubscriptExp(self, n: ASTSubscriptExp):
        val_ptr = self.visit(n.val)
        slicing_args = []
        for s in n.slicing.data:
            if isinstance(s, ASTExpression):
                slicing_args.append(self.visit(s))
            elif isinstance(s, Tuple):
                slicing_args.append(self._ir_builder.op_parenthesis([self.visit(s[0]), self.visit(s[1]), self.visit(s[2])]))
        return self._ir_builder.op_get_item(val_ptr, self._ir_builder.op_square_brackets(slicing_args), dbg=n.dbg)

    def visit_ASTLoad(self, n: ASTLoad):
        val_ptr = self._ir_ctx.get(n.name)
        if val_ptr is None:
            raise VariableNotFoundError(n.dbg, f'Variable {n.name} referenced but not defined.')
        return val_ptr

    def visit_ASTSquareBrackets(self, n: ASTSquareBrackets):
        values = []
        for value in n.values:
            if isinstance(value, ASTStarredExpr):
                iter_values = self._ir_builder.op_iter(self.visit(value.value), value.value.dbg)
                values.extend(iter_values.values())
            else:
                values.append(self.visit(value))
        return self._ir_builder.op_square_brackets(values, dbg=n.dbg)

    def visit_ASTParenthesis(self, n: ASTParenthesis):
        values = []
        for value in n.values:
            if isinstance(value, ASTStarredExpr):
                iter_values = self._ir_builder.op_iter(self.visit(value.value), value.value.dbg)
                values.extend(iter_values.values())
            else:
                values.append(self.visit(value))
        return self._ir_builder.op_parenthesis(values, dbg=n.dbg)

    def visit_ASTGeneratorExp(self, n: ASTGeneratorExp):
        generated_expressions = []
        def _for_each_generator(generators: List[ASTGenerator]):
            gen = generators[0]
            iter_exp = gen.iter
            iter_elts = self._ir_builder.op_iter(self.visit(iter_exp))
            for loop_index_ptr in iter_elts.values():
                self._do_recursive_assign(gen.target, loop_index_ptr, False)
                cond = self._ir_builder.ir_constant_int(1)
                for _if in gen.ifs:
                    cond = self._ir_builder.ir_logical_and(cond, self._ir_builder.op_bool_cast(self.visit(_if)))
                if cond.val() is None:
                    raise StaticInferenceError(n.dbg, "Cannot statically infer the condition value in the generator expression. This is crucial to determine the datatype of the generated expression.")
                if not cond.val():
                    continue
                if len(generators) == 1:
                    generated_expressions.append(self.visit(n.elt))
                else:
                    _for_each_generator(generators[1:])

        self._ir_ctx.generator_enter()
        _for_each_generator(n.generators)
        self._ir_ctx.generator_leave()
        if n.kind == ASTGeneratorExp.Kind.LIST:
            return self._ir_builder.op_square_brackets(generated_expressions, dbg=n.dbg)
        return self._ir_builder.op_parenthesis(generated_expressions, dbg=n.dbg)

    def visit_ASTCondExp(self, n: ASTCondExp):
        cond_ptr = self.visit(n.cond)
        true_expr = self.visit(n.t_expr)
        false_expr = self.visit(n.f_expr)
        cond_ptr = self._ir_builder.op_bool_cast(cond_ptr, dbg=n.dbg)
        return self._ir_builder.op_select(cond_ptr, true_expr, false_expr)

    def visit_ASTJoinedStr(self, n: ASTJoinedStr):
        values = [self.visit(val) for val in n.values]
        str_value = self._ir_builder.ir_constant_str("")
        for val in values:
            str_value = self._ir_builder.op_add(str_value, val)
        return str_value

    def visit_ASTFormattedValue(self, n: ASTFormattedValue):
        return self._ir_builder.op_str(self.visit(n.value))

    def visit_ASTStarredExpr(self, n: ASTStarredExpr):
        raise UnpackingError(n.dbg, "Can't use starred expression here.")

    def visit_chip_call(self, chip: InternalChipObject, args: List[Value], kwargs: Dict[str, Value]):
        chip_declared_args = [(x.name, x.annotation) for x in chip.chip_ast.inputs]
        chip_filled_args = [None for _ in range(len(chip_declared_args))]
        if self._ir_ctx.get_recursion_depth() >= self._config.recursion_limit():
            self._ir_builder.op_assert(
                self._ir_builder.ir_constant_int(0),
                self._ir_builder.ir_logical_and(self._ir_ctx.get_condition_value_for_assertion(), self._ir_ctx.get_condition_value()),
                dbg=chip.chip_ast.dbg
            )
            # TODO: raise a warning here
            return self._ir_builder.op_placeholder_value(chip.return_dt, chip.chip_ast.dbg)
        for i, arg in enumerate(args):
            if i >= len(chip_declared_args):
                raise ChipArgumentsError(chip.chip_ast.dbg, "Too many chip arguments")
            arg_name, arg_anno = chip_declared_args[i]
            if arg_anno is not None and arg_anno.dt != arg.type():
                raise ChipArgumentsError(chip.chip_ast.dbg, f"Chip argument datatype mismatch for `{arg_name}`")
            chip_filled_args[i] = arg
        for key, arg in kwargs.items():
            arg_assigned = False
            for i, (name, anno) in enumerate(chip_declared_args):
                if name == key:
                    if chip_filled_args[i] is not None:
                        raise ChipArgumentsError(chip.chip_ast.dbg, f"Chip argument `{name}` already assigned")
                    if anno is not None and anno.dt != arg.type():
                        raise ChipArgumentsError(chip.chip_ast.dbg, f"Chip argument datatype mismatch for `{name}`")
                    chip_filled_args[i] = arg
                    arg_assigned = True
                    break
            if not arg_assigned:
                raise ChipArgumentsError(chip.chip_ast.dbg, f"No such argument `{key}`")
        for i, (name, anno) in enumerate(chip_declared_args):
            if chip_filled_args[i] is None:
                raise ChipArgumentsError(chip.chip_ast.dbg, f"Chip argument for `{name}` not filled")
        self._ir_ctx.add_recursion_depth()
        self._ir_ctx.chip_enter(chip.return_dt, self._ir_ctx.get_condition_value())
        self._register_global_datatypes()
        for i, (name, anno) in enumerate(chip_declared_args):
            self._ir_ctx.set(name, chip_filled_args[i])
        for i, stmt in enumerate(chip.chip_ast.block):
            self.visit(stmt)
        return_vals_cond = self._ir_ctx.get_returns_with_conditions()
        if not self._ir_ctx.check_return_guaranteed() and not isinstance(chip.return_dt, NoneDTDescriptor):
            raise ControlEndWithoutReturnError(chip.chip_ast.dbg, "Chip control ends without a return statement")
        self._ir_ctx.chip_leave()
        self._ir_ctx.sub_recursion_depth()
        if isinstance(chip.return_dt, NoneDTDescriptor):
            return self._ir_builder.op_constant_none()
        return_value = return_vals_cond[-1][0]
        for val, cond in reversed(return_vals_cond[:-1]):
            return_value = self._ir_builder.op_select(cond, val, return_value)
        return return_value

    def visit_external_call(self, external_func: InternalExternalFuncObject, args: List[Value], kwargs: Dict[str, Value]):
        external_call_id = self._next_external_call_id
        self._next_external_call_id += 1
        for i, arg in enumerate(args):
            self._ir_builder.op_export_external(arg, external_call_id, i, ())
        for key, arg in kwargs.items():
            self._ir_builder.op_export_external(arg, external_call_id, key, ())
        self._ir_builder.ir_invoke_external(
            external_call_id, external_func.name,
            [arg.type() for arg in args], {key: v.type() for key, v in kwargs.items()}
        )
        return self._ir_builder.op_input((external_call_id, ), external_func.return_dt, "Public")

    def _register_global_datatypes(self):
        float_class = self._ir_builder.op_constant_class(FloatDTDescriptor())
        integer_class = self._ir_builder.op_constant_class(IntegerDTDescriptor())
        for name in FloatDTDescriptor.get_alise_typenames():
            self._ir_ctx.set(name, float_class)
        for name in IntegerDTDescriptor.get_alise_typenames():
            self._ir_ctx.set(name, integer_class)

    def _do_recursive_assign(self, target: ASTAssignTarget, value: Value, conditional_select: bool):
        if isinstance(target, ASTNameAssignTarget):
            if self._ir_ctx.exists(target.name):
                orig_value = self._ir_ctx.get(target.name)
                if orig_value.type_locked() and value.type() != orig_value.type():
                    raise InterScopeError(target.dbg, f"Cannot assign to `{target.name}`: this variable is declared at the outer scope. Attempting to change its datatype in the inner scope from {orig_value.type()} to {value.type()} is not allowed. Assigning to variables from outer scope must keep its datatype and shape.")
                if orig_value.type_locked() and conditional_select:
                    value = self._ir_builder.op_select(self._ir_ctx.get_condition_value(), value, orig_value, dbg=target.dbg)
            self._ir_ctx.set(target.name, value)
        elif isinstance(target, ASTSubscriptAssignTarget):
            target_value = self.visit(target.target)
            slicing_args = []
            for s in target.slicing.data:
                if isinstance(s, ASTExpression):
                    slicing_args.append(self.visit(s))
                elif isinstance(s, Tuple):
                    slicing_args.append(self._ir_builder.op_parenthesis([self.visit(s[0]), self.visit(s[1]), self.visit(s[2])]))
            slicing_args_value = self._ir_builder.op_square_brackets(slicing_args)
            # we don't care conditional_select here. it is always true
            assert conditional_select
            value = self._ir_builder.op_set_item(self._ir_ctx.get_condition_value(), target_value, slicing_args_value, value, dbg=target.dbg)
        elif isinstance(target, ASTListAssignTarget) or isinstance(target, ASTTupleAssignTarget):
            assert sum([1 for x in target.targets if x.star]) <= 1
            has_star = any([x.star for x in target.targets])
            elements = self._ir_builder.op_iter(value).values()
            if len(elements) < len(target.targets):
                raise UnpackingError(target.dbg, f"Not enough elements to unpack, expected {len(target.targets)} got {len(elements)}")
            if len(elements) > len(target.targets) and not has_star:
                raise UnpackingError(target.dbg, f"Too many elements to unpack, expected {len(target.targets)} got {len(elements)}")
            if has_star:
                star_idx = [i for i, x in enumerate(target.targets) if x.star][0]
                elements_for_star = elements[star_idx:len(elements) - (len(target.targets) - star_idx - 1)]
                for i, tgt in enumerate(target.targets[:star_idx]):
                    self._do_recursive_assign(tgt, elements[i], conditional_select)
                for i, tgt in enumerate(target.targets[star_idx + 1:]):
                    self._do_recursive_assign(tgt, elements[star_idx + len(elements_for_star) + i], conditional_select)
                if len(elements_for_star) == 1:
                    self._do_recursive_assign(target.targets[star_idx], elements_for_star[0], conditional_select)
                else:
                    self._do_recursive_assign(target.targets[star_idx], self._ir_builder.op_square_brackets(list(elements_for_star)), conditional_select)
            else:
                for i, tgt in enumerate(target.targets):
                    self._do_recursive_assign(tgt, elements[i], conditional_select)
