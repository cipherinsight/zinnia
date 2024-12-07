from typing import List, Tuple, Dict

from pyzk.ir.ir_pass.constant_fold import ConstantFoldIRPass
from pyzk.ir.ir_pass.dead_code_elimination import DeadCodeEliminationIRPass
from pyzk.ir.ir_pass.expose_public_inserter import ExposePublicInserterIRPass
from pyzk.ir.ir_pass.input_metadata_extractor import InputMetadataExtractorIRPass
from pyzk.ir.ir_pass.ndarray_flattener import NDArrayFlattenerIRPass
from pyzk.ir.ir_stmt import IRStatement
from pyzk.ast.zk_ast import ASTComponent, ASTProgram, ASTAssignStatement, ASTPassStatement, ASTSlicingAssignStatement, \
    ASTExpression, ASTForInStatement, ASTCondStatement, ASTConstant, \
    ASTSlicing, ASTLoad, ASTAssertStatement, ASTSlicingData, ASTSquareBrackets, ASTBreakStatement, ASTContinueStatement, \
    ASTBinaryOperator, ASTNamedAttribute, ASTExprAttribute, ASTParenthesis, ASTChip, ASTReturnStatement, \
    ASTCallStatement
from pyzk.ir.ir_builder import IRBuilder
from pyzk.ir.ir_ctx import IRContext
from pyzk.debug.exception import VariableNotFoundError, ConstantInferenceError, NoForElementsError, NotInLoopError, \
    InterScopeError, UnsupportedLangFeatureException, UnreachableStatementError, OperatorOrChipNotFoundException, \
    StatementNoEffectException, ControlEndWithoutReturnError, ReturnDatatypeMismatchError, \
    ChipArgumentsError
from pyzk.opdef.operator_factory import Operators
from pyzk.internal.dt_descriptor import DTDescriptorFactory, NoneDTDescriptor
from pyzk.internal.prog_meta_data import ProgramMetadata
from pyzk.internal.annotation import Annotation
from pyzk.algo.ndarray_helper import NDArrayHelper
from pyzk.debug.dbg_info import DebugInfo


class IRGenerator:
    def __init__(self):
        self._ir_ctx = IRContext()
        self._ir_builder = IRBuilder(self._ir_ctx)

    def generate(self, component: ASTComponent) -> Tuple[List[IRStatement], ProgramMetadata]:
        prog_meta_data = ProgramMetadata()
        self.visit(component)
        ir_graph = self._ir_builder.export_ir_graph()
        ir_graph = InputMetadataExtractorIRPass(prog_meta_data).exec(ir_graph)
        ir_graph = ExposePublicInserterIRPass().exec(ir_graph)
        ir_graph = NDArrayFlattenerIRPass().exec(ir_graph)
        ir_graph = ConstantFoldIRPass().exec(ir_graph)
        ir_graph = DeadCodeEliminationIRPass().exec(ir_graph)
        return ir_graph.export_stmts(), prog_meta_data

    def visit(self, component: ASTComponent):
        typename = type(component).__name__
        method_name = 'visit_' + typename
        method = getattr(self, method_name, None)
        if method is None:
            raise NotImplementedError(method_name)
        return method(component)

    def visit_chip(self, chip: ASTChip, args: List[int], kwargs: Dict[str, int]):
        chip_declared_args = [(x.name, x.annotation.dt) for x in chip.inputs]
        chip_filled_args = [None for x in range(len(chip_declared_args))]
        for i, arg in enumerate(args):
            if i >= len(chip_declared_args):
                raise ChipArgumentsError(chip.dbg_i, "Too many chip arguments")
            if chip_declared_args[i][1] != self._ir_ctx.get_inferred_datatype(arg):
                raise ChipArgumentsError(chip.dbg_i, f"Chip argument datatype mismatch for `{chip_declared_args[i][0]}`")
            chip_filled_args[i] = arg
        for key, arg in kwargs.items():
            arg_assigned = False
            for i, (name, dt) in enumerate(chip_declared_args):
                if name == key:
                    if chip_filled_args[i] is not None:
                        raise ChipArgumentsError(chip.dbg_i, f"Chip argument `{name}` already assigned")
                    if dt != self._ir_ctx.get_inferred_datatype(arg):
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
        for i, (name, dt) in enumerate(chip_declared_args):
            self._ir_ctx.assign_name_to_ptr(name, chip_filled_args[i])
        for i, stmt in enumerate(chip.block):
            self.visit(stmt)
        return_vals_cond = self._ir_ctx.get_return_values_and_conditions()
        if not self._ir_ctx.get_branch_has_default_return() and not isinstance(chip.return_anno.dt, NoneDTDescriptor):
            raise ControlEndWithoutReturnError(chip.dbg_i, "Chip control ends without a return statement")
        self._ir_ctx.chip_leave()
        if isinstance(chip.return_anno.dt, NoneDTDescriptor):
            return self._ir_builder.create_constant_none()
        return_value = return_vals_cond[-1][0]
        for val, cond in reversed(return_vals_cond[:-1]):
            return_value = self._ir_builder.create_select(cond, val, return_value)
        return return_value

    def visit_ASTProgram(self, n: ASTProgram):
        for i, inp in enumerate(n.inputs):
            ptr = self._ir_builder.create_input(
                i, inp.annotation.dt, inp.annotation.public,
                dbg_i=n.dbg_i, annotation=Annotation(inp.annotation.dt, inp.annotation.public)
            )
            self._ir_ctx.assign_name_to_ptr(inp.name, ptr)
        for key, ast in n.chips.items():
            self._ir_ctx.register_chip(key, ast)
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
            elif not self._ir_ctx.is_inferred_datatype_equal(val_ptr, orig_val_ptr):
                raise InterScopeError(n.dbg_i, f"Cannot assign to `{n.assignee}`: this variable is declared at the outer scope. Attempting to change its datatype in the inner scope from {self._ir_ctx.get_inferred_datatype_name(orig_val_ptr)} to {self._ir_ctx.get_inferred_datatype_name(val_ptr)} is not allowed. Assigning to variables from outer scope must keep its datatype and shape.")
            else:
                val_ptr = self._create_assignment_with_condition(
                    orig_val_ptr, val_ptr, dbg_i=n.dbg_i, annotation=annotation)
        self._ir_ctx.assign_name_to_ptr(n.assignee, val_ptr)
        return val_ptr

    def visit_ASTSlicingAssignStatement(self, n: ASTSlicingAssignStatement):
        if self._ir_ctx.get_branch_has_default_return():
            raise UnreachableStatementError(n.dbg_i, "This code is unreachable")
        val_ptr = self.visit(n.value)
        orig_val_ptr = self._ir_ctx.lookup_ptr_by_name(n.assignee)
        assert orig_val_ptr is not None
        assert n.annotation is None
        val_ptr = self._create_assignment_with_condition(orig_val_ptr, val_ptr, dbg_i=n.dbg_i, annotation=None)
        val_ptr = self._ir_builder.create_slicing_assign([self._as_constant_slicing(sli) for sli in n.slicing.data], orig_val_ptr, val_ptr, dbg_i=n.dbg_i, annotation=None)
        self._ir_ctx.assign_name_to_ptr(n.assignee, val_ptr)
        return val_ptr

    def visit_ASTForInStatement(self, n: ASTForInStatement):
        if self._ir_ctx.get_branch_has_default_return():
            raise UnreachableStatementError(n.dbg_i, "This code is unreachable")
        iter_expr_ptr = self.visit(n.iter_expr)
        iter_elts = self._as_constant_ndarray(n.iter_expr)
        backup_ptr = self._ir_ctx.lookup_ptr_by_name(n.assignee)
        if len(iter_elts) == 0:
            raise NoForElementsError(n.dbg_i, "No iterable elements found in the for statement.")
        self._ir_ctx.for_block_enter()
        for i in range(len(iter_elts)):
            loop_index_ptr = self._ir_builder.create_slicing(iter_expr_ptr, [(i, )], dbg_i=n.dbg_i)
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
        condition_result = self._ir_builder.create_constant(1)
        for var in condition_vars:
            condition_result = self._ir_builder.create_logical_and(condition_result, var)
        condition_result = self._ir_builder.create_logical_not(condition_result)
        self._ir_ctx.for_block_break(condition_result)
        return None

    def visit_ASTContinueStatement(self, n: ASTContinueStatement):
        if self._ir_ctx.get_branch_has_default_return():
            raise UnreachableStatementError(n.dbg_i, "This code is unreachable")
        if not self._ir_ctx.for_block_exists():
            raise NotInLoopError(n.dbg_i, "Invalid continue statement here outside the loop.")
        condition_vars = self._ir_ctx.get_condition_variables()
        condition_result = self._ir_builder.create_constant(1)
        for var in condition_vars:
            condition_result = self._ir_builder.create_logical_and(condition_result, var)
        condition_result = self._ir_builder.create_logical_not(condition_result)
        self._ir_ctx.for_block_continue(condition_result)
        return None

    def visit_ASTCondStatement(self, n: ASTCondStatement):
        if self._ir_ctx.get_branch_has_default_return():
            raise UnreachableStatementError(n.dbg_i, "This code is unreachable")
        cond_ptr = self.visit(n.cond)
        true_cond_ptr = self._ir_builder.create_bool_cast(cond_ptr, dbg_i=n.dbg_i)
        false_cond_ptr = self._ir_builder.create_logical_not(true_cond_ptr, dbg_i=n.dbg_i)
        has_default_return_t_block = has_default_return_f_block = False
        if self._ir_ctx.get_inferred_constant_value(true_cond_ptr) != 0:
            self._ir_ctx.if_block_enter(true_cond_ptr)
            self._ir_ctx.block_enter()
            for _, stmt in enumerate(n.t_block):
                self.visit(stmt)
            self._ir_ctx.block_leave()
            has_default_return_t_block = self._ir_ctx.get_branch_has_default_return()
            self._ir_ctx.if_block_leave()
        if self._ir_ctx.get_inferred_constant_value(false_cond_ptr) != 0:
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
        return self._ir_builder.create_assert(test_wrt_conditions, dbg_i=n.dbg_i)

    def visit_ASTReturnStatement(self, n: ASTReturnStatement):
        if self._ir_ctx.get_branch_has_default_return():
            raise UnreachableStatementError(n.dbg_i, "Unreachable return statement")
        if self._ir_ctx.get_chip_depth() == 0:
            raise UnsupportedLangFeatureException(n.dbg_i, "Return statements in circuits is not supported")
        val = self.visit(n.expr)
        if val is None:
            val = self.visit(n.expr)
        return_dt = self._ir_ctx.get_inferred_datatype(val)
        expected_return_dt = self._ir_ctx.get_current_chip().return_anno.dt
        if return_dt != expected_return_dt:
            raise ReturnDatatypeMismatchError(n.dbg_i, "Return datatype mismatch annotated return datatype")
        return_condition = self._ir_builder.create_constant(1)
        for condition in self._ir_ctx.get_condition_variables():
            return_condition = self._ir_builder.create_logical_and(return_condition, condition)
        return_prevent_condition = self._ir_builder.create_logical_not(return_condition)
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
        kwargs = n.operator.params_parse(
            n.dbg_i,
            [self.visit(arg) for arg in n.args],
            {k: self.visit(arg) for k, arg in n.kwargs.items()}
        )
        return self._ir_builder.create_op(n.operator, kwargs, dbg_i=n.dbg_i)

    def visit_ASTUnaryOperator(self, n: ASTBinaryOperator):
        kwargs = n.operator.params_parse(
            n.dbg_i,
            [self.visit(arg) for arg in n.args],
            {k: self.visit(arg) for k, arg in n.kwargs.items()}
        )
        return self._ir_builder.create_op(n.operator, kwargs, dbg_i=n.dbg_i)

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
            dt = self._ir_ctx.get_inferred_datatype(target_ptr)
            operator = Operators.instantiate_operator(n.member, dt.get_typename())
            self_arg = [target_ptr]
            if operator is None:
                raise OperatorOrChipNotFoundException(n.dbg_i, f"Operator {n.target}::{n.member} not found")
        kwargs = operator.params_parse(
            n.dbg_i,
            self_arg + visited_args,
            visited_kwargs
        )
        return self._ir_builder.create_op(operator, kwargs, dbg_i=n.dbg_i)

    def visit_ASTExprAttribute(self, n: ASTExprAttribute):
        target_ptr = self.visit(n.target)
        dt = self._ir_ctx.get_inferred_datatype(target_ptr)
        operator = Operators.instantiate_operator(n.member, dt.get_typename())
        kwargs = operator.params_parse(
            n.dbg_i,
            [target_ptr] + [self.visit(arg) for arg in n.args],
            {k: self.visit(arg) for k, arg in n.kwargs.items()}
        )
        return self._ir_builder.create_op(operator, kwargs, dbg_i=n.dbg_i)

    def visit_ASTConstant(self, n: ASTConstant):
        return self._ir_builder.create_constant(n.value, dbg_i=n.dbg_i)

    def visit_ASTSlicing(self, n: ASTSlicing):
        val_ptr = self.visit(n.val)
        return self._ir_builder.create_slicing(val_ptr, self._as_constant_slicing(n.slicing), dbg_i=n.dbg_i)

    def visit_ASTLoad(self, n: ASTLoad):
        val_ptr = self._ir_ctx.lookup_ptr_by_name(n.name)
        if val_ptr is None:
            raise VariableNotFoundError(n.dbg_i, f'Variable {n.name} referenced but not defined.')
        return val_ptr

    def visit_ASTSquareBrackets(self, n: ASTSquareBrackets):
        values = [self.visit(val) for val in n.values]
        return self._ir_builder.create_square_brackets(values, dbg_i=n.dbg_i)

    def visit_ASTParenthesis(self, n: ASTParenthesis):
        values = [self.visit(val) for val in n.values]
        return self._ir_builder.create_parenthesis(values, dbg_i=n.dbg_i)

    def _create_assignment_with_condition(self, orig_val_ptr, new_val_ptr, dbg_i: DebugInfo = None, annotation: Annotation | None = None):
        cond_stack = self._ir_ctx.get_condition_variables()
        if len(cond_stack) == 0:
            return new_val_ptr
        cond_val_ptr = cond_stack[0]
        for cond in cond_stack[1:]:
            cond_val_ptr = self._ir_builder.create_logical_and(cond_val_ptr, cond, dbg_i=dbg_i)
        return self._ir_builder.create_select(cond_val_ptr, new_val_ptr, orig_val_ptr, dbg_i=dbg_i)

    def _create_assert_with_condition(self, expr_val_ptr, dbg_i: DebugInfo = None, annotation: Annotation | None = None):
        cond_stack = self._ir_ctx.get_condition_variables()
        if len(cond_stack) == 0:
            return expr_val_ptr
        cond_val_ptr = cond_stack[0]
        for cond in cond_stack[1:]:
            cond_val_ptr = self._ir_builder.create_logical_and(cond_val_ptr, cond, dbg_i=dbg_i)
        constant_1 = self._ir_builder.create_constant(1)
        return self._ir_builder.create_select(cond_val_ptr, expr_val_ptr, constant_1, dbg_i=dbg_i)

    def _as_constant_integer(self, n: ASTExpression) -> int:
        ptr = self.visit(n)
        result = self._ir_ctx.get_inferred_constant_value(ptr)
        if result is None:
            raise ConstantInferenceError(n.dbg_i, "Cannot infer the corresponding constant value for this expression. Please make sure that here should be a constant scalar number.")
        if not isinstance(result, int):
            raise ConstantInferenceError(n.dbg_i, "This is expression inferred as a constant ndarray. Please make sure that here should be a constant scalar number.")
        return result

    def _as_constant_ndarray(self, n: ASTExpression) -> List:
        ptr = self.visit(n)
        result = self._ir_ctx.get_inferred_constant_value(ptr)
        if result is None:
            raise ConstantInferenceError(n.dbg_i, "Cannot infer the corresponding constant value for this expression. Please make sure that here should be a constant ndarray.")
        if not isinstance(result, NDArrayHelper):
            raise ConstantInferenceError(n.dbg_i, "This is expression inferred as a constant scalar number. Please make sure that here should be a constant ndarray.")
        return result.values

    def _as_constant_slicing(self, n: ASTSlicingData) -> List[Tuple[int, ...]]:
        results = []
        for data in n.data:
            if isinstance(data, ASTExpression):
                val = self._as_constant_integer(data)
                results.append((val, ))
            else:
                if data[0] is not None:
                    val_l = self._as_constant_integer(data[0])
                else:
                    val_l = None
                if data[1] is not None:
                    val_r = self._as_constant_integer(data[1])
                else:
                    val_r = None
                if data[2] is not None:
                    val_s = self._as_constant_integer(data[2])
                else:
                    val_s = None
                results.append((val_l, val_r, val_s))
        return results
