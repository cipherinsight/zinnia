from typing import List, Tuple, Dict

from zenopy.builder.builder_impl import IRBuilderImpl
from zenopy.builder.value import Value
from zenopy.ir.ir_pass.always_satisfied_elimination import AlwaysSatisfiedEliminationIRPass
from zenopy.ir.ir_pass.constant_fold import ConstantFoldIRPass
from zenopy.ir.ir_pass.dead_code_elimination import DeadCodeEliminationIRPass
from zenopy.ir.ir_pass.duplicate_code_elimination import DuplicateCodeEliminationIRPass
from zenopy.ir.ir_stmt import IRStatement
from zenopy.ast.zk_ast import ASTComponent, ASTProgram, ASTAssignStatement, ASTPassStatement, ASTSlicingAssignStatement, \
    ASTExpression, ASTForInStatement, ASTCondStatement, ASTConstantFloat, ASTConstantInteger, \
    ASTSlicing, ASTLoad, ASTAssertStatement, ASTSquareBrackets, ASTBreakStatement, ASTContinueStatement, \
    ASTBinaryOperator, ASTNamedAttribute, ASTExprAttribute, ASTParenthesis, ASTChip, ASTReturnStatement, \
    ASTCallStatement, ASTUnaryOperator, ASTConstantNone
from zenopy.ir.ir_ctx import IRContext
from zenopy.debug.exception import VariableNotFoundError, NoForElementsError, NotInLoopError, \
    InterScopeError, UnsupportedLangFeatureException, UnreachableStatementError, OperatorOrChipNotFoundException, \
    StatementNoEffectException, ControlEndWithoutReturnError, ReturnDatatypeMismatchError, \
    ChipArgumentsError
from zenopy.opdef.operator_factory import Operators
from zenopy.internal.dt_descriptor import DTDescriptorFactory, NoneDTDescriptor, FloatDTDescriptor, IntegerDTDescriptor
from zenopy.internal.prog_meta_data import ProgramMetadata, ProgramInputMetadata
from zenopy.internal.annotation import Annotation
from zenopy.debug.dbg_info import DebugInfo


class IRGenerator:
    def __init__(self):
        self._ir_ctx = IRContext()
        self._ir_builder = IRBuilderImpl(self._ir_ctx)
        self.prog_meta_data = None

    def generate(self, component: ASTComponent) -> Tuple[List[IRStatement], ProgramMetadata]:
        self.prog_meta_data = ProgramMetadata()
        self.visit(component)
        ir_graph = self._ir_builder.export_ir_graph()
        ir_graph = ConstantFoldIRPass().exec(ir_graph)
        ir_graph = DeadCodeEliminationIRPass().exec(ir_graph)
        ir_graph = AlwaysSatisfiedEliminationIRPass().exec(ir_graph)
        ir_graph = DuplicateCodeEliminationIRPass().exec(ir_graph)
        return ir_graph.export_stmts(), self.prog_meta_data

    def visit(self, component: ASTComponent):
        typename = type(component).__name__
        method_name = 'visit_' + typename
        method = getattr(self, method_name, None)
        if method is None:
            raise NotImplementedError(method_name)
        return method(component)

    def visit_chip(self, chip: ASTChip, args: List[Value], kwargs: Dict[str, Value]):
        chip_declared_args = [(x.name, x.annotation.dt) for x in chip.inputs]
        chip_filled_args = [None for x in range(len(chip_declared_args))]
        for i, arg in enumerate(args):
            if i >= len(chip_declared_args):
                raise ChipArgumentsError(chip.dbg_i, "Too many chip arguments")
            if chip_declared_args[i][1] != arg.type():
                raise ChipArgumentsError(chip.dbg_i, f"Chip argument datatype mismatch for `{chip_declared_args[i][0]}`")
            chip_filled_args[i] = arg
        for key, arg in kwargs.items():
            arg_assigned = False
            for i, (name, dt) in enumerate(chip_declared_args):
                if name == key:
                    if chip_filled_args[i] is not None:
                        raise ChipArgumentsError(chip.dbg_i, f"Chip argument `{name}` already assigned")
                    if dt != arg.type():
                        raise ChipArgumentsError(chip.dbg_i, f"Chip argument datatype mismatch for `{name}`")
                    chip_filled_args[i] = arg
                    arg_assigned = True
                    break
            if not arg_assigned:
                raise ChipArgumentsError(chip.dbg_i, f"No such argument `{key}`")
        for i, (name, dt) in enumerate(chip_declared_args):
            if chip_filled_args[i] is None:
                raise ChipArgumentsError(chip.dbg_i, f"Chip argument for `{name}` not filled")
        self._ir_ctx.chip_enter(chip)
        self._register_global_datatypes()
        for i, (name, dt) in enumerate(chip_declared_args):
            self._ir_ctx.assign_name_to_ptr(name, chip_filled_args[i])
        for i, stmt in enumerate(chip.block):
            self.visit(stmt)
        return_vals_cond = self._ir_ctx.get_return_values_and_conditions()
        if not self._ir_ctx.get_branch_has_default_return() and not isinstance(chip.return_anno.dt, NoneDTDescriptor):
            raise ControlEndWithoutReturnError(chip.dbg_i, "Chip control ends without a return statement")
        self._ir_ctx.chip_leave()
        if isinstance(chip.return_anno.dt, NoneDTDescriptor):
            return self._ir_builder.op_constant_none()
        return_value = return_vals_cond[-1][0]
        for val, cond in reversed(return_vals_cond[:-1]):
            return_value = self._ir_builder.op_select(cond, val, return_value)
        return return_value

    def visit_ASTProgram(self, n: ASTProgram):
        for i, inp in enumerate(n.inputs):
            ptr = self._ir_builder.op_input(
                i, inp.annotation.dt, inp.annotation.public,
                dbg=n.dbg_i
            )
            self._ir_ctx.assign_name_to_ptr(inp.name, ptr)
            self.prog_meta_data.inputs.append(ProgramInputMetadata(inp.annotation.dt, inp.name, inp.annotation.public))
        for key, ast in n.chips.items():
            self._ir_ctx.register_chip(key, ast)
        self._register_global_datatypes()
        for i, stmt in enumerate(n.block):
            self.visit(stmt)
        return None

    def visit_ASTPassStatement(self, n: ASTPassStatement):
        if self._ir_ctx.get_branch_has_default_return():
            raise UnreachableStatementError(n.dbg_i, "This code is unreachable")
        return None

    def visit_ASTAssignStatement(self, n: ASTAssignStatement):
        if self._ir_ctx.get_branch_has_default_return():
            raise UnreachableStatementError(n.dbg_i, "This code is unreachable")
        val_ptr = self.visit(n.value)
        orig_val_ptr = self._ir_ctx.lookup_ptr_by_name(n.assignee)
        annotation = None
        if n.annotation is not None:
            annotation = Annotation(n.annotation.dt, n.annotation.public)
        if orig_val_ptr is not None:
            if self._ir_ctx.is_name_in_stack_scope_top(n.assignee):
                pass
            elif val_ptr.type() != orig_val_ptr.type():
                raise InterScopeError(n.dbg_i, f"Cannot assign to `{n.assignee}`: this variable is declared at the outer scope. Attempting to change its datatype in the inner scope from {orig_val_ptr.type()} to {val_ptr.type()} is not allowed. Assigning to variables from outer scope must keep its datatype and shape.")
            else:
                val_ptr = self._create_assignment_with_condition(
                    orig_val_ptr, val_ptr, dbg_i=n.dbg_i, annotation=annotation)
        self._ir_ctx.assign_name_to_ptr(n.assignee, val_ptr)
        return val_ptr

    def visit_ASTSlicingAssignStatement(self, n: ASTSlicingAssignStatement):
        if self._ir_ctx.get_branch_has_default_return():
            raise UnreachableStatementError(n.dbg_i, "This code is unreachable")
        val_ptr = self.visit(n.value)
        orig_val_ptr = self.visit(n.assignee)
        slicing_args = []
        for s in n.slicing.data:
            if isinstance(s, ASTExpression):
                slicing_args.append(self.visit(s))
            elif isinstance(s, Tuple):
                slicing_args.append((self.visit(s[0]), self.visit(s[1]), self.visit(s[2])))
        val_ptr = self._create_assignment_with_condition(orig_val_ptr, val_ptr, dbg_i=n.dbg_i, annotation=None)
        val_ptr = self._ir_builder.op_set_item(orig_val_ptr,
            self._ir_builder.op_square_brackets(slicing_args), val_ptr, dbg=n.dbg_i)
        return val_ptr

    def visit_ASTForInStatement(self, n: ASTForInStatement):
        if self._ir_ctx.get_branch_has_default_return():
            raise UnreachableStatementError(n.dbg_i, "This code is unreachable")
        iter_elts = self._ir_builder.op_iter(self.visit(n.iter_expr))
        if not all(x == iter_elts.types()[0] for x in iter_elts.types()):
            raise NoForElementsError(n.dbg_i, "In for statement, all elements in the iterable must have the same type")
        backup_ptr = self._ir_ctx.lookup_ptr_by_name(n.assignee)
        if len(iter_elts.values()) == 0:
            raise NoForElementsError(n.dbg_i, "No iterable elements found in the for statement.")
        self._ir_ctx.for_block_enter()
        for loop_index_ptr in iter_elts.values():
            self._ir_ctx.assign_name_to_ptr(n.assignee, loop_index_ptr)
            self._ir_ctx.block_enter()
            self._ir_ctx.for_block_reiter()
            for _, stmt in enumerate(n.block):
                self.visit(stmt)
            self._ir_ctx.block_leave()
        self._ir_ctx.for_block_leave()
        if backup_ptr is not None:
            self._ir_ctx.assign_name_to_ptr(n.assignee, backup_ptr)
        return None

    def visit_ASTBreakStatement(self, n: ASTBreakStatement):
        if self._ir_ctx.get_branch_has_default_return():
            raise UnreachableStatementError(n.dbg_i, "This code is unreachable")
        if not self._ir_ctx.for_block_exists():
            raise NotInLoopError(n.dbg_i, "Invalid break statement here outside the loop.")
        condition_vars = self._ir_ctx.get_condition_variables()
        condition_result = self._ir_builder.ir_constant_int(1)
        for var in condition_vars:
            condition_result = self._ir_builder.ir_logical_and(condition_result, var)
        condition_result = self._ir_builder.ir_logical_not(condition_result)
        self._ir_ctx.for_block_break(condition_result)
        return None

    def visit_ASTContinueStatement(self, n: ASTContinueStatement):
        if self._ir_ctx.get_branch_has_default_return():
            raise UnreachableStatementError(n.dbg_i, "This code is unreachable")
        if not self._ir_ctx.for_block_exists():
            raise NotInLoopError(n.dbg_i, "Invalid continue statement here outside the loop.")
        condition_vars = self._ir_ctx.get_condition_variables()
        condition_result = self._ir_builder.ir_constant_int(1)
        for var in condition_vars:
            condition_result = self._ir_builder.ir_logical_and(condition_result, var)
        condition_result = self._ir_builder.ir_logical_not(condition_result)
        self._ir_ctx.for_block_continue(condition_result)
        return None

    def visit_ASTCondStatement(self, n: ASTCondStatement):
        if self._ir_ctx.get_branch_has_default_return():
            raise UnreachableStatementError(n.dbg_i, "This code is unreachable")
        cond_ptr = self.visit(n.cond)
        true_cond_ptr = self._ir_builder.op_bool_scalar(cond_ptr, dbg=n.dbg_i)
        false_cond_ptr = self._ir_builder.ir_logical_not(true_cond_ptr, dbg=n.dbg_i)
        has_default_return_t_block = has_default_return_f_block = False
        self._ir_ctx.if_block_enter(true_cond_ptr)
        self._ir_ctx.block_enter()
        for _, stmt in enumerate(n.t_block):
            self.visit(stmt)
        self._ir_ctx.block_leave()
        has_default_return_t_block = self._ir_ctx.get_branch_has_default_return()
        self._ir_ctx.if_block_leave()
        self._ir_ctx.if_block_enter(false_cond_ptr)
        self._ir_ctx.block_enter()
        for _, stmt in enumerate(n.f_block):
            self.visit(stmt)
        self._ir_ctx.block_leave()
        has_default_return_f_block = self._ir_ctx.get_branch_has_default_return()
        self._ir_ctx.if_block_leave()
        if has_default_return_t_block and has_default_return_f_block:
            assert not self._ir_ctx.get_branch_has_default_return()
            self._ir_ctx.set_branch_has_default_return()
        return None

    def visit_ASTAssertStatement(self, n: ASTAssertStatement):
        if self._ir_ctx.get_branch_has_default_return():
            raise UnreachableStatementError(n.dbg_i, "This code is unreachable")
        test = self.visit(n.expr)
        test_wrt_conditions = self._create_assert_with_condition(test, n.dbg_i)
        return self._ir_builder.op_assert(test_wrt_conditions, dbg=n.dbg_i)

    def visit_ASTReturnStatement(self, n: ASTReturnStatement):
        if self._ir_ctx.get_branch_has_default_return():
            raise UnreachableStatementError(n.dbg_i, "Unreachable return statement")
        if self._ir_ctx.get_chip_depth() == 0:
            raise UnsupportedLangFeatureException(n.dbg_i, "Return statements in circuits is not supported")
        if n.expr is not None:
            val = self.visit(n.expr)
            return_dt = val.type()
        else:
            val = self._ir_builder.op_constant_none()
            return_dt = NoneDTDescriptor()
        expected_return_dt = self._ir_ctx.get_current_chip().return_anno.dt
        if return_dt != expected_return_dt:
            raise ReturnDatatypeMismatchError(n.dbg_i, "Return datatype mismatch annotated return datatype")
        return_condition = self._ir_builder.ir_constant_int(1)
        for condition in self._ir_ctx.get_condition_variables():
            return_condition = self._ir_builder.ir_logical_and(return_condition, condition)
        return_prevent_condition = self._ir_builder.ir_logical_not(return_condition)
        self._ir_ctx.add_return_value(val, return_condition, return_prevent_condition)
        return None

    def visit_ASTCallStatement(self, n: ASTCallStatement):
        if self._ir_ctx.get_branch_has_default_return():
            raise UnreachableStatementError(n.dbg_i, "This code is unreachable")
        visited_args = [self.visit(arg) for arg in n.args]
        visited_kwargs = {k: self.visit(arg) for k, arg in n.kwargs.items()}
        chip = self._ir_ctx.lookup_chip(n.name)
        if chip is not None:
            return self.visit_chip(chip, visited_args, visited_kwargs)
        if Operators.instantiate_operator(n.name, None) is not None:
            raise StatementNoEffectException(n.dbg_i, f"Statement seems to have no effect")
        raise OperatorOrChipNotFoundException(n.dbg_i, f"Chip {n.name} not found")

    def visit_ASTBinaryOperator(self, n: ASTBinaryOperator):
        lhs_expr = self.visit(n.lhs)
        rhs_expr = self.visit(n.rhs)
        if n.operator == n.Op.ADD:
            return self._ir_builder.op_add(lhs_expr, rhs_expr, dbg=n.dbg_i)
        elif n.operator == n.Op.SUB:
            return self._ir_builder.op_subtract(lhs_expr, rhs_expr, dbg=n.dbg_i)
        elif n.operator == n.Op.MUL:
            return self._ir_builder.op_multiply(lhs_expr, rhs_expr, dbg=n.dbg_i)
        elif n.operator == n.Op.DIV:
            return self._ir_builder.op_divide(lhs_expr, rhs_expr, dbg=n.dbg_i)
        elif n.operator == n.Op.MOD:
            return self._ir_builder.op_modulo(lhs_expr, rhs_expr, dbg=n.dbg_i)
        elif n.operator == n.Op.FLOOR_DIV:
            return self._ir_builder.op_floor_divide(lhs_expr, rhs_expr, dbg=n.dbg_i)
        elif n.operator == n.Op.MAT_MUL:
            return self._ir_builder.op_mat_mul(lhs_expr, rhs_expr, dbg=n.dbg_i)
        elif n.operator == n.Op.EQ:
            return self._ir_builder.op_equal(lhs_expr, rhs_expr, dbg=n.dbg_i)
        elif n.operator == n.Op.NE:
            return self._ir_builder.op_not_equal(lhs_expr, rhs_expr, dbg=n.dbg_i)
        elif n.operator == n.Op.LT:
            return self._ir_builder.op_less_than(lhs_expr, rhs_expr, dbg=n.dbg_i)
        elif n.operator == n.Op.LTE:
            return self._ir_builder.op_less_than_or_equal(lhs_expr, rhs_expr, dbg=n.dbg_i)
        elif n.operator == n.Op.GT:
            return self._ir_builder.op_greater_than(lhs_expr, rhs_expr, dbg=n.dbg_i)
        elif n.operator == n.Op.GTE:
            return self._ir_builder.op_greater_than_or_equal(lhs_expr, rhs_expr, dbg=n.dbg_i)
        elif n.operator == n.Op.AND:
            return self._ir_builder.ir_logical_and(lhs_expr, rhs_expr, dbg=n.dbg_i)
        elif n.operator == n.Op.OR:
            return self._ir_builder.ir_logical_or(lhs_expr, rhs_expr, dbg=n.dbg_i)
        raise NotImplementedError(f"Internal Error: Binary Operator {n.operator} not implemented")

    def visit_ASTUnaryOperator(self, n: ASTUnaryOperator):
        operand_expr = self.visit(n.operand)
        if n.operator == n.Op.NOT:
            return self._ir_builder.op_unary_not(operand_expr, dbg=n.dbg_i)
        if n.operator == n.Op.USUB:
            return self._ir_builder.op_unary_sub(operand_expr, dbg=n.dbg_i)
        raise NotImplementedError(f"Internal Error: Unary Operator {n.operator} not implemented")

    def visit_ASTNamedAttribute(self, n: ASTNamedAttribute):
        self_arg = []
        visited_args = [self.visit(arg) for arg in n.args]
        visited_kwargs = {k: self.visit(arg) for k, arg in n.kwargs.items()}
        if n.target is None or DTDescriptorFactory.is_typename(n.target):
            chip = self._ir_ctx.lookup_chip(n.member)
            if n.target is None and chip is not None:
                return self.visit_chip(chip, visited_args, visited_kwargs)
            operator = Operators.instantiate_operator(n.member, n.target)
            if operator is None:
                raise OperatorOrChipNotFoundException(n.dbg_i, f"Operator or Chip {n.member} not found")
        else:
            target_ptr = self._ir_ctx.lookup_ptr_by_name(n.target)
            if target_ptr is None:
                raise VariableNotFoundError(n.dbg_i, f'Variable {n.target} referenced but not defined.')
            dt = target_ptr.type()
            operator = Operators.instantiate_operator(n.member, dt.get_typename())
            self_arg = [target_ptr]
            if operator is None:
                raise OperatorOrChipNotFoundException(n.dbg_i, f"Operator {n.target}::{n.member} not found")
        return self._ir_builder.invoke_op(operator, self_arg + visited_args, visited_kwargs, dbg=n.dbg_i)

    def visit_ASTExprAttribute(self, n: ASTExprAttribute):
        target_ptr = self.visit(n.target)
        dt = target_ptr.type()
        operator = Operators.instantiate_operator(n.member, dt.get_typename())
        return self._ir_builder.invoke_op(
            operator,
            [target_ptr] + [self.visit(arg) for arg in n.args],
            {k: self.visit(arg) for k, arg in n.kwargs.items()},
            dbg=n.dbg_i
        )

    def visit_ASTConstantFloat(self, n: ASTConstantFloat):
        return self._ir_builder.ir_constant_float(n.value, dbg=n.dbg_i)

    def visit_ASTConstantInteger(self, n: ASTConstantInteger):
        return self._ir_builder.ir_constant_int(n.value, dbg=n.dbg_i)

    def visit_ASTConstantNone(self, n: ASTConstantNone):
        return self._ir_builder.op_constant_none(dbg=n.dbg_i)

    def visit_ASTSlicing(self, n: ASTSlicing):
        val_ptr = self.visit(n.val)
        slicing_args = []
        for s in n.slicing.data:
            if isinstance(s, ASTExpression):
                slicing_args.append(self.visit(s))
            elif isinstance(s, Tuple):
                slicing_args.append(self._ir_builder.op_parenthesis([self.visit(s[0]), self.visit(s[1]), self.visit(s[2])]))
        return self._ir_builder.op_get_item(val_ptr, self._ir_builder.op_square_brackets(slicing_args), dbg=n.dbg_i)

    def visit_ASTLoad(self, n: ASTLoad):
        val_ptr = self._ir_ctx.lookup_ptr_by_name(n.name)
        if val_ptr is None:
            raise VariableNotFoundError(n.dbg_i, f'Variable {n.name} referenced but not defined.')
        return val_ptr

    def visit_ASTSquareBrackets(self, n: ASTSquareBrackets):
        values = [self.visit(val) for val in n.values]
        return self._ir_builder.op_square_brackets(values, dbg=n.dbg_i)

    def visit_ASTParenthesis(self, n: ASTParenthesis):
        values = [self.visit(val) for val in n.values]
        return self._ir_builder.op_parenthesis(values, dbg=n.dbg_i)

    def _register_global_datatypes(self):
        float_class = self._ir_builder.op_constant_class(FloatDTDescriptor())
        integer_class = self._ir_builder.op_constant_class(IntegerDTDescriptor())
        self._ir_ctx.assign_name_to_ptr("Float", float_class)
        self._ir_ctx.assign_name_to_ptr("Integer", integer_class)

    def _create_assignment_with_condition(self, orig_val_ptr, new_val_ptr, dbg_i: DebugInfo = None, annotation: Annotation | None = None):
        cond_stack = self._ir_ctx.get_condition_variables()
        if len(cond_stack) == 0:
            return new_val_ptr
        cond_val_ptr = cond_stack[0]
        for cond in cond_stack[1:]:
            cond_val_ptr = self._ir_builder.ir_logical_and(cond_val_ptr, cond, dbg=dbg_i)
        return self._ir_builder.op_select(cond_val_ptr, new_val_ptr, orig_val_ptr, dbg=dbg_i)

    def _create_assert_with_condition(self, expr_val_ptr, dbg_i: DebugInfo = None, annotation: Annotation | None = None):
        cond_stack = self._ir_ctx.get_condition_variables()
        if len(cond_stack) == 0:
            return expr_val_ptr
        cond_val_ptr = cond_stack[0]
        for cond in cond_stack[1:]:
            cond_val_ptr = self._ir_builder.ir_logical_and(cond_val_ptr, cond, dbg=dbg_i)
        constant_1 = self._ir_builder.ir_constant_int(1)
        return self._ir_builder.op_select(cond_val_ptr, expr_val_ptr, constant_1, dbg=dbg_i)
