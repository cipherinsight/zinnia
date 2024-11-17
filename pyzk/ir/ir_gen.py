from typing import List, Tuple

from .ir_pass.constant_fold import ConstantFoldIRPass
from .ir_pass.dead_code_elimination import DeadCodeEliminationIRPass
from .ir_pass.expose_public_inserter import ExposePublicInserterIRPass
from .ir_pass.input_metadata_extractor import InputMetadataExtractorIRPass
from .ir_pass.ndarray_flattener import NDArrayFlattenerIRPass
from .ir_stmt import IRStatement
from ..ast.zk_ast import ASTComponent, ASTProgram, ASTAssignStatement, ASTPassStatement, ASTSlicingAssignStatement, \
    ASTExpression, ASTForInStatement, ASTCondStatement, ASTOperator, ASTConstant, \
    ASTSlicing, ASTLoad, ASTAssertStatement, ASTSlicingData, ASTCreateNDArray, ASTBreakStatement, ASTContinueStatement
from .ir_builder import IRBuilder
from .ir_ctx import IRContext
from ..exception.contextual import VariableNotFoundError, ConstantInferenceError, NoForElementsError, NotInLoopError, \
    InterScopeError
from pyzk.util.prog_meta_data import ProgramMetadata
from ..util.annotation import Annotation
from ..util.ndarray_helper import NDArrayHelper
from ..util.op_name import OpName
from ..util.source_pos_info import SourcePosInfo


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
        ir_graph = NDArrayFlattenerIRPass(prog_meta_data).exec(ir_graph)
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

    def visit_ASTProgram(self, n: ASTProgram):
        for i, inp in enumerate(n.inputs):
            ptr = self._ir_builder.create_input(i, source_pos_info=n.source_pos_info, annotation=Annotation(
                inp.annotation.typename, inp.annotation.shape, inp.public))
            self._ir_ctx.assign_name_to_ptr(inp.name, ptr)
        for i, stmt in enumerate(n.block):
            self.visit(stmt)
        return None

    def visit_ASTPassStatement(self, n: ASTPassStatement):
        return None

    def visit_ASTAssignStatement(self, n: ASTAssignStatement):
        val_ptr = self.visit(n.value)
        orig_val_ptr = self._ir_ctx.lookup_ptr_by_name(n.assignee)
        annotation = None
        if n.annotation is not None:
            annotation = Annotation(n.annotation.typename, n.annotation.shape, n.annotation.public)
        if orig_val_ptr is not None:
            if self._ir_ctx.is_name_in_stack_scope_top(n.assignee):
                pass
            elif not self._ir_ctx.is_inferred_datatype_equal(val_ptr, orig_val_ptr):
                raise InterScopeError(n.source_pos_info, f"Cannot assign to `{n.assignee}`: this variable is declared at the outer scope. Attempting to change its datatype in the inner scope from {self._ir_ctx.get_inferred_datatype_name(orig_val_ptr)} to {self._ir_ctx.get_inferred_datatype_name(val_ptr)} is not allowed. Assigning to variables from outer scope must keep its datatype and shape.")
            else:
                val_ptr = self._create_assignment_with_condition(
                    orig_val_ptr, val_ptr, source_pos_info=n.source_pos_info, annotation=annotation)
        self._ir_ctx.assign_name_to_ptr(n.assignee, val_ptr)
        return val_ptr

    def visit_ASTSlicingAssignStatement(self, n: ASTSlicingAssignStatement):
        val_ptr = self.visit(n.value)
        orig_val_ptr = self._ir_ctx.lookup_ptr_by_name(n.assignee)
        assert orig_val_ptr is not None
        assert n.annotation is None
        val_ptr = self._create_assignment_with_condition(orig_val_ptr, val_ptr, source_pos_info=n.source_pos_info, annotation=None)
        val_ptr = self._ir_builder.create_slicing_assign([self._as_constant_slicing(sli) for sli in n.slicing.data], orig_val_ptr, val_ptr, source_pos_info=n.source_pos_info, annotation=None)
        self._ir_ctx.assign_name_to_ptr(n.assignee, val_ptr)
        return val_ptr

    def visit_ASTForInStatement(self, n: ASTForInStatement):
        iter_expr_ptr = self.visit(n.iter_expr)
        iter_elts = self._as_constant_ndarray(n.iter_expr)
        backup_ptr = self._ir_ctx.lookup_ptr_by_name(n.assignee)
        if len(iter_elts) == 0:
            raise NoForElementsError(n.source_pos_info, "No iterable elements found in the for statement.")
        self._ir_ctx.for_block_enter(self._ir_builder.create_constant(1), self._ir_builder.create_constant(0))
        for i in range(len(iter_elts)):
            loop_index_ptr = self._ir_builder.create_slicing(iter_expr_ptr, [i], source_pos_info=n.source_pos_info)
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
        if not self._ir_ctx.for_block_exists():
            raise NotInLoopError(n.source_pos_info, "Invalid break statement here outside the loop.")
        self._ir_ctx.for_block_break()
        return None

    def visit_ASTContinueStatement(self, n: ASTContinueStatement):
        if not self._ir_ctx.for_block_exists():
            raise NotInLoopError(n.source_pos_info, "Invalid continue statement here outside the loop.")
        self._ir_ctx.for_block_continue()
        return None

    def visit_ASTCondStatement(self, n: ASTCondStatement):
        cond_ptr = self.visit(n.cond)
        true_cond_ptr = self._ir_builder.create_is_true(cond_ptr, source_pos_info=n.source_pos_info)
        false_cond_ptr = self._ir_builder.create_is_false(cond_ptr, source_pos_info=n.source_pos_info)
        self._ir_ctx.if_block_enter(true_cond_ptr)
        self._ir_ctx.block_enter()
        for _, stmt in enumerate(n.t_block):
            self.visit(stmt)
        self._ir_ctx.block_leave()
        self._ir_ctx.if_block_leave()
        self._ir_ctx.if_block_enter(false_cond_ptr)
        self._ir_ctx.block_enter()
        for _, stmt in enumerate(n.f_block):
            self.visit(stmt)
        self._ir_ctx.block_leave()
        self._ir_ctx.if_block_leave()
        return None

    def visit_ASTAssertStatement(self, n: ASTAssertStatement):
        test = self.visit(n.expr)
        test_wrt_conditions = self._create_assert_with_condition(test, n.source_pos_info)
        return self._ir_builder.create_assert(test_wrt_conditions, source_pos_info=n.source_pos_info)

    def visit_ASTOperator(self, n: ASTOperator):
        op_name = n.op
        args = []
        constant_args = None
        if OpName.is_constant_operator(op_name):
            constant_args = list([self._as_constant_integer(arg) for arg in n.args])
        elif OpName.is_constant_arg_operator(op_name):
            args = [self.visit(n.args[0])]
            constant_args = list([self._as_constant_integer(arg) for arg in n.args[1:]])
        else:
            args = [self.visit(arg) for arg in n.args]
        return self._ir_builder.create_op(op_name, args, constant_args, source_pos_info=n.source_pos_info)

    def visit_ASTConstant(self, n: ASTConstant):
        return self._ir_builder.create_constant(n.value, source_pos_info=n.source_pos_info)

    def visit_ASTSlicing(self, n: ASTSlicing):
        val_ptr = self.visit(n.val)
        return self._ir_builder.create_slicing(val_ptr, self._as_constant_slicing(n.slicing), source_pos_info=n.source_pos_info)

    def visit_ASTLoad(self, n: ASTLoad):
        val_ptr = self._ir_ctx.lookup_ptr_by_name(n.name)
        if val_ptr is None:
            raise VariableNotFoundError(n.source_pos_info, f'Variable {n.name} referenced but not defined.')
        return val_ptr

    def visit_ASTCreateNDArray(self, n: ASTCreateNDArray):
        values = [self.visit(val) for val in n.values]
        return self._ir_builder.create_new_list(values, source_pos_info=n.source_pos_info)

    def _create_assignment_with_condition(self, orig_val_ptr, new_val_ptr, source_pos_info: SourcePosInfo = None, annotation: Annotation | None = None):
        cond_stack = self._ir_ctx.get_condition_variables()
        if len(cond_stack) == 0:
            return new_val_ptr
        cond_val_ptr = cond_stack[0]
        for cond in cond_stack[1:]:
            cond_val_ptr = self._ir_builder.create_logical_and(cond_val_ptr, cond, source_pos_info=source_pos_info)
        orelse_cond_val_ptr = self._ir_builder.create_logical_not(cond_val_ptr, source_pos_info=source_pos_info)
        val_1 = self._ir_builder.create_mul(cond_val_ptr, new_val_ptr, source_pos_info=source_pos_info)
        val_2 = self._ir_builder.create_mul(orelse_cond_val_ptr, orig_val_ptr, source_pos_info=source_pos_info)
        return self._ir_builder.create_add(val_1, val_2, source_pos_info=source_pos_info, annotation=annotation)

    def _create_assert_with_condition(self, expr_val_ptr, source_pos_info: SourcePosInfo = None, annotation: Annotation | None = None):
        cond_stack = self._ir_ctx.get_condition_variables()
        if len(cond_stack) == 0:
            return expr_val_ptr
        cond_val_ptr = cond_stack[0]
        for cond in cond_stack[1:]:
            cond_val_ptr = self._ir_builder.create_logical_and(cond_val_ptr, cond, source_pos_info=source_pos_info)
        orelse_cond_val_ptr = self._ir_builder.create_logical_not(cond_val_ptr, source_pos_info=source_pos_info)
        val_1 = self._ir_builder.create_mul(cond_val_ptr, expr_val_ptr, source_pos_info=source_pos_info)
        return self._ir_builder.create_add(val_1, orelse_cond_val_ptr, source_pos_info=source_pos_info, annotation=annotation)

    def _as_constant_integer(self, n: ASTExpression) -> int:
        ptr = self.visit(n)
        result = self._ir_ctx.get_inferred_constant_value(ptr)
        if result is None:
            raise ConstantInferenceError(n.source_pos_info, "Cannot infer the corresponding constant value for this expression. Please make sure that here should be a constant scalar number.")
        if not isinstance(result, int):
            raise ConstantInferenceError(n.source_pos_info, "This is expression inferred as a constant ndarray. Please make sure that here should be a constant scalar number.")
        return result

    def _as_constant_ndarray(self, n: ASTExpression) -> List:
        ptr = self.visit(n)
        result = self._ir_ctx.get_inferred_constant_value(ptr)
        if result is None:
            raise ConstantInferenceError(n.source_pos_info, "Cannot infer the corresponding constant value for this expression. Please make sure that here should be a constant ndarray.")
        if not isinstance(result, NDArrayHelper):
            raise ConstantInferenceError(n.source_pos_info, "This is expression inferred as a constant scalar number. Please make sure that here should be a constant ndarray.")
        return result.values

    def _as_constant_slicing(self, n: ASTSlicingData) -> List[Tuple[int, int, int] | int]:
        results = []
        for data in n.data:
            if isinstance(data, ASTExpression):
                val = self._as_constant_integer(data)
                results.append(val)
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
