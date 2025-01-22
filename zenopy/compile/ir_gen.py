import copy
from typing import List, Tuple, Dict

from zenopy.ast.zk_ast_tree import ZKAbstractSyntaxTree
from zenopy.builder.builder_impl import IRBuilderImpl
from zenopy.builder.value import Value, HashedValue
from zenopy.compile.multi_pass.always_satisfied_elimination import AlwaysSatisfiedEliminationIRPass
from zenopy.compile.multi_pass.constant_fold import ConstantFoldIRPass
from zenopy.compile.multi_pass.dead_code_elimination import DeadCodeEliminationIRPass
from zenopy.compile.multi_pass.duplicate_code_elimination import DuplicateCodeEliminationIRPass
from zenopy.compile.ir_stmt import IRStatement
from zenopy.ast.zk_ast import ASTComponent, ASTProgram, ASTAssignStatement, ASTPassStatement, \
    ASTExpression, ASTForInStatement, ASTCondStatement, ASTConstantFloat, ASTConstantInteger, \
    ASTSlicing, ASTLoad, ASTAssertStatement, ASTSquareBrackets, ASTBreakStatement, ASTContinueStatement, \
    ASTBinaryOperator, ASTNamedAttribute, ASTExprAttribute, ASTParenthesis, ASTChip, ASTReturnStatement, \
    ASTCallStatement, ASTUnaryOperator, ASTConstantNone, ASTNameAssignTarget, ASTSubscriptAssignTarget, \
    ASTListAssignTarget, ASTTupleAssignTarget, ASTAssignTarget, ASTExprStatement, ASTConstantString, ASTGeneratorExp, \
    ASTGenerator, ASTCondExp, ASTProgramInput
from zenopy.compile.ir_ctx import IRContext
from zenopy.compile.multi_pass.external_call_remover import ExternalCallRemoverIRPass
from zenopy.debug.exception import VariableNotFoundError, NoForElementsError, NotInLoopError, \
    InterScopeError, UnsupportedLangFeatureException, UnreachableStatementError, OperatorOrChipNotFoundException, \
    StatementNoEffectException, ControlEndWithoutReturnError, ReturnDatatypeMismatchError, \
    ChipArgumentsError, TupleUnpackingError, StaticInferenceError
from zenopy.internal.external_call import ExternalCall
from zenopy.internal.external_func_obj import ExternalFuncObj
from zenopy.opdef.ir_op.ir_read_float import ReadFloatIR
from zenopy.opdef.ir_op.ir_read_integer import ReadIntegerIR
from zenopy.opdef.operator_factory import Operators
from zenopy.internal.dt_descriptor import DTDescriptorFactory, NoneDTDescriptor, FloatDTDescriptor, IntegerDTDescriptor, \
    IntegerType, FloatType
from zenopy.internal.prog_meta_data import ProgramMetadata, ProgramInputMetadata, ProgramCompiledInputMetadata
from zenopy.debug.dbg_info import DebugInfo


class IRGenerator:
    def __init__(self):
        self._ir_builder = IRBuilderImpl()
        self._ir_ctx = IRContext(self._ir_builder)
        self._prog_meta_data = None
        self._registered_chips = {}
        self._registered_externals = {}
        self._next_external_call_id = 1
        self._external_calls = []

    def generate(self, zk_ast: ZKAbstractSyntaxTree) -> Tuple[List[IRStatement], List[IRStatement], List[ExternalCall], ProgramMetadata]:
        self._registered_chips = zk_ast.chips
        self._registered_externals = zk_ast.externals
        self.visit(zk_ast.root)
        ir_graph = self._ir_builder.export_ir_graph()
        ir_graph = ExternalCallRemoverIRPass().exec(ir_graph)
        # ir_graph = ConstantFoldIRPass().exec(ir_graph)
        # ir_graph = DeadCodeEliminationIRPass().exec(ir_graph)
        # ir_graph = AlwaysSatisfiedEliminationIRPass().exec(ir_graph)
        # ir_graph = DuplicateCodeEliminationIRPass().exec(ir_graph)
        ir_stmts = ir_graph.export_stmts()
        ir_graph_for_preprocess = self._ir_builder.export_ir_graph()
        # ir_graph_for_preprocess = ConstantFoldIRPass().exec(ir_graph_for_preprocess)
        # ir_graph_for_preprocess = DeadCodeEliminationIRPass().exec(ir_graph_for_preprocess)
        # ir_graph_for_preprocess = AlwaysSatisfiedEliminationIRPass().exec(ir_graph_for_preprocess)
        # ir_graph_for_preprocess = DuplicateCodeEliminationIRPass().exec(ir_graph_for_preprocess)
        ir_stmts_for_preprocess = ir_graph_for_preprocess.export_stmts()
        self._prog_meta_data = ProgramMetadata()
        self._prog_meta_data.set_program_inputs([ProgramInputMetadata(inp.annotation.dt, inp.name, inp.annotation.kind) for inp in zk_ast.root.inputs])
        compiled_inputs = []
        for stmt in ir_stmts:
            if isinstance(stmt.operator, ReadIntegerIR):
                compiled_inputs.append(ProgramCompiledInputMetadata(IntegerType, stmt.operator.indices))
            elif isinstance(stmt.operator, ReadFloatIR):
                compiled_inputs.append(ProgramCompiledInputMetadata(FloatType, stmt.operator.indices))
        self._prog_meta_data.set_program_compiled_inputs(compiled_inputs)
        return ir_stmts, ir_stmts_for_preprocess, self._external_calls, self._prog_meta_data

    def visit(self, component: ASTComponent):
        typename = type(component).__name__
        method_name = 'visit_' + typename
        method = getattr(self, method_name, None)
        if method is None:
            raise NotImplementedError(method_name)
        return method(component)

    def visit_ASTProgram(self, n: ASTProgram):
        for i, inp in enumerate(n.inputs):
            ptr = self._ir_builder.op_input((0, i), inp.annotation.dt, inp.annotation.kind, dbg=n.dbg)
            self._ir_ctx.set(inp.name, ptr.val() if isinstance(ptr, HashedValue) else ptr)
        self._register_global_datatypes()
        for i, stmt in enumerate(n.block):
            self.visit(stmt)
        return None

    def visit_ASTPassStatement(self, n: ASTPassStatement):
        if self._ir_ctx.check_return_existence():
            raise UnreachableStatementError(n.dbg, "This code is unreachable")
        return None

    def visit_ASTAssignStatement(self, n: ASTAssignStatement):
        if self._ir_ctx.check_return_existence():
            raise UnreachableStatementError(n.dbg, "This code is unreachable")
        val_ptr = self.visit(n.value)
        for target in n.targets:
            self._do_recursive_assign(target, val_ptr, True)
        return val_ptr

    def visit_ASTForInStatement(self, n: ASTForInStatement):
        if self._ir_ctx.check_return_existence():
            raise UnreachableStatementError(n.dbg, "This code is unreachable")
        iter_elts = self._ir_builder.op_iter(self.visit(n.iter_expr))
        self._ir_ctx.loop_enter()
        for loop_index_ptr in iter_elts.values():
            self._do_recursive_assign(n.target, loop_index_ptr, False)
            self._ir_ctx.loop_reiter()
            for _, stmt in enumerate(n.block):
                self.visit(stmt)
        self._ir_ctx.loop_leave()
        return None

    def visit_ASTBreakStatement(self, n: ASTBreakStatement):
        if self._ir_ctx.check_return_existence():
            raise UnreachableStatementError(n.dbg, "This code is unreachable")
        if not self._ir_ctx.is_in_loop():
            raise NotInLoopError(n.dbg, "Invalid break statement here outside the loop.")
        self._ir_ctx.loop_break()
        return None

    def visit_ASTContinueStatement(self, n: ASTContinueStatement):
        if self._ir_ctx.check_return_existence():
            raise UnreachableStatementError(n.dbg, "This code is unreachable")
        if not self._ir_ctx.is_in_loop():
            raise NotInLoopError(n.dbg, "Invalid continue statement here outside the loop.")
        self._ir_ctx.loop_continue()
        return None

    def visit_ASTCondStatement(self, n: ASTCondStatement):
        if self._ir_ctx.check_return_existence():
            raise UnreachableStatementError(n.dbg, "This code is unreachable")
        cond_ptr = self.visit(n.cond)
        true_cond_ptr = self._ir_builder.op_bool_scalar(cond_ptr, dbg=n.dbg)
        false_cond_ptr = self._ir_builder.ir_logical_not(true_cond_ptr, dbg=n.dbg)
        self._ir_ctx.if_enter(true_cond_ptr)
        for _, stmt in enumerate(n.t_block):
            self.visit(stmt)
        scope_true = self._ir_ctx.if_leave()
        self._ir_ctx.if_enter(false_cond_ptr)
        for _, stmt in enumerate(n.f_block):
            self.visit(stmt)
        scope_false = self._ir_ctx.if_leave()
        if scope_true.has_return_statement() and scope_false.has_return_statement():
            self._ir_ctx.set_has_return()
        return None

    def visit_ASTAssertStatement(self, n: ASTAssertStatement):
        if self._ir_ctx.check_return_existence():
            raise UnreachableStatementError(n.dbg, "This code is unreachable")
        test = self.visit(n.expr)
        return self._ir_builder.op_assert(test, self._ir_ctx.get_condition_value(), dbg=n.dbg)

    def visit_ASTReturnStatement(self, n: ASTReturnStatement):
        if self._ir_ctx.check_return_existence():
            raise UnreachableStatementError(n.dbg, "Unreachable return statement")
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
        self._ir_ctx.return_value(val)
        return None

    def visit_ASTCallStatement(self, n: ASTCallStatement):
        if self._ir_ctx.check_return_existence():
            raise UnreachableStatementError(n.dbg, "This code is unreachable")
        visited_args = [self.visit(arg) for arg in n.args]
        visited_kwargs = {k: self.visit(arg) for k, arg in n.kwargs.items()}
        chip = self._registered_chips.get(n.name, None)
        if chip is not None:
            return self.visit_chip_call(chip, visited_args, visited_kwargs)
        external_func = self._registered_externals.get(n.name, None)
        if external_func is not None:
            return self.visit_external_call(external_func, visited_args, visited_kwargs)
        if Operators.instantiate_operator(n.name, None) is not None:
            raise StatementNoEffectException(n.dbg, f"Statement seems to have no effect")
        raise OperatorOrChipNotFoundException(n.dbg, f"Chip {n.name} not found")

    def visit_ASTExprStatement(self, n: ASTExprStatement):
        if self._ir_ctx.check_return_existence():
            raise UnreachableStatementError(n.dbg, "This code is unreachable")
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
            return self._ir_builder.op_power(lhs_expr, rhs_expr, None, dbg=n.dbg)
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
        visited_args = [self.visit(arg) for arg in n.args]
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
                raise OperatorOrChipNotFoundException(n.dbg, f"Operator or Chip {n.member} not found")
        else:
            target_ptr = self._ir_ctx.get(n.target)
            if target_ptr is None:
                raise VariableNotFoundError(n.dbg, f'Variable {n.target} referenced but not defined.')
            dt = target_ptr.type()
            operator = Operators.instantiate_operator(n.member, dt.get_typename())
            self_arg = [target_ptr]
            if operator is None:
                raise OperatorOrChipNotFoundException(n.dbg, f"Operator {n.target}::{n.member} not found")
        return self._ir_builder.create_op(operator, self_arg + visited_args, visited_kwargs, dbg=n.dbg)

    def visit_ASTExprAttribute(self, n: ASTExprAttribute):
        target_ptr = self.visit(n.target)
        dt = target_ptr.type()
        operator = Operators.instantiate_operator(n.member, dt.get_typename())
        return self._ir_builder.create_op(
            operator,
            [target_ptr] + [self.visit(arg) for arg in n.args],
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
        return self._ir_builder.op_constant_string(n.value, dbg=n.dbg)

    def visit_ASTSlicing(self, n: ASTSlicing):
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
        values = [self.visit(val) for val in n.values]
        return self._ir_builder.op_square_brackets(values, dbg=n.dbg)

    def visit_ASTParenthesis(self, n: ASTParenthesis):
        values = [self.visit(val) for val in n.values]
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
                    cond = self._ir_builder.ir_logical_and(cond, self._ir_builder.op_bool_scalar(self.visit(_if)))
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
        cond_ptr = self._ir_builder.op_bool_scalar(cond_ptr, dbg=n.dbg)
        return self._ir_builder.op_select(cond_ptr, true_expr, false_expr)

    def visit_chip_call(self, chip: ASTChip, args: List[Value], kwargs: Dict[str, Value]):
        chip_declared_args = [(x.name, x.annotation) for x in chip.inputs]
        chip_filled_args = [None for _ in range(len(chip_declared_args))]
        for i, arg in enumerate(args):
            if i >= len(chip_declared_args):
                raise ChipArgumentsError(chip.dbg, "Too many chip arguments")
            arg_name, arg_anno = chip_declared_args[i]
            if arg_anno is not None and arg_anno.dt != arg.type():
                raise ChipArgumentsError(chip.dbg, f"Chip argument datatype mismatch for `{arg_name}`")
            chip_filled_args[i] = arg
        for key, arg in kwargs.items():
            arg_assigned = False
            for i, (name, anno) in enumerate(chip_declared_args):
                if name == key:
                    if chip_filled_args[i] is not None:
                        raise ChipArgumentsError(chip.dbg, f"Chip argument `{name}` already assigned")
                    if anno is not None and anno.dt != arg.type():
                        raise ChipArgumentsError(chip.dbg, f"Chip argument datatype mismatch for `{name}`")
                    chip_filled_args[i] = arg
                    arg_assigned = True
                    break
            if not arg_assigned:
                raise ChipArgumentsError(chip.dbg, f"No such argument `{key}`")
        for i, (name, anno) in enumerate(chip_declared_args):
            if chip_filled_args[i] is None:
                raise ChipArgumentsError(chip.dbg, f"Chip argument for `{name}` not filled")
        self._ir_ctx.chip_enter(chip.return_anno.dt)
        self._register_global_datatypes()
        for i, (name, anno) in enumerate(chip_declared_args):
            self._ir_ctx.set(name, chip_filled_args[i])
        for i, stmt in enumerate(chip.block):
            self.visit(stmt)
        return_vals_cond = self._ir_ctx.get_returns_with_conditions()
        if not self._ir_ctx.check_return_existence() and not isinstance(chip.return_anno.dt, NoneDTDescriptor):
            raise ControlEndWithoutReturnError(chip.dbg, "Chip control ends without a return statement")
        self._ir_ctx.chip_leave()
        if isinstance(chip.return_anno.dt, NoneDTDescriptor):
            return self._ir_builder.op_constant_none()
        return_value = return_vals_cond[-1][0]
        for val, cond in reversed(return_vals_cond[:-1]):
            return_value = self._ir_builder.op_select(cond, val, return_value)
        return return_value

    def visit_external_call(self, external_func: ExternalFuncObj, args: List[Value], kwargs: Dict[str, Value]):
        external_call_id = self._next_external_call_id
        self._next_external_call_id += 1
        for i, arg in enumerate(args):
            self._ir_builder.op_export_external(arg, external_call_id, i, ())
        for key, arg in kwargs.items():
            self._ir_builder.op_export_external(arg, external_call_id, key, ())
        self._ir_builder.ir_invoke_external(external_call_id)
        self._external_calls.append(ExternalCall(
            external_call_id, external_func.name,
            [arg.type() for arg in args], {key: v.type() for key, v in kwargs.items()})
        )
        return self._ir_builder.op_input((external_call_id, ), external_func.return_dt, "Public")

    def _register_global_datatypes(self):
        float_class = self._ir_builder.op_constant_class(FloatDTDescriptor())
        integer_class = self._ir_builder.op_constant_class(IntegerDTDescriptor())
        self._ir_ctx.set("Float", float_class)
        self._ir_ctx.set("Integer", integer_class)

    def _do_recursive_assign(self, target: ASTAssignTarget, value: Value, conditional_select: bool):
        if isinstance(target, ASTNameAssignTarget):
            if self._ir_ctx.exists(target.name):
                orig_value = self._ir_ctx.get(target.name)
                if not self._ir_ctx.exists_in_top_scope(target.name) and value.type() != orig_value.type():
                    raise InterScopeError(target.dbg, f"Cannot assign to `{target.name}`: this variable is declared at the outer scope. Attempting to change its datatype in the inner scope from {orig_value.type()} to {value.type()} is not allowed. Assigning to variables from outer scope must keep its datatype and shape.")
                if not self._ir_ctx.exists_in_top_scope(target.name) and conditional_select:
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
            if conditional_select:
                orig_value = self._ir_builder.op_get_item(target_value, slicing_args_value, dbg=target.dbg)
                value = self._ir_builder.op_select(self._ir_ctx.get_condition_value(), value, orig_value, dbg=target.dbg)
            value = self._ir_builder.op_set_item(target_value, slicing_args_value, value, dbg=target.dbg)
        elif isinstance(target, ASTListAssignTarget) or isinstance(target, ASTTupleAssignTarget):
            assert sum([1 for x in target.targets if x.star]) <= 1
            has_star = any([x.star for x in target.targets])
            elements = self._ir_builder.op_iter(value).values()
            if len(elements) < len(target.targets):
                raise TupleUnpackingError(target.dbg, f"Not enough elements to unpack, expected {len(target.targets)} got {len(elements)}")
            if len(elements) > len(target.targets) and not has_star:
                raise TupleUnpackingError(target.dbg, f"Too many elements to unpack, expected {len(target.targets)} got {len(elements)}")
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
