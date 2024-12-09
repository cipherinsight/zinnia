from typing import Tuple, List, Dict

from pyzk.ir.ir_ctx import IRContext
from pyzk.ir.ir_graph import IRGraph, IRGraphMetadata
from pyzk.ir.ir_stmt import IRStatement
from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.internal.annotation import Annotation
from pyzk.internal.dt_descriptor import DTDescriptor
from pyzk.opdef.operator_factory import Operators
from pyzk.debug.dbg_info import DebugInfo


def _check_annotation_and_raise(lhs: Annotation | None, rhs: Annotation | None):
    if lhs is None or rhs is None:
        return
    # TODO
    

class IRBuilder:
    def __init__(self, ir_ctx: IRContext | None = None) -> None:
        self.stmts = []
        self._next_id = len(self.stmts)
        self.ir_ctx = ir_ctx

    def create_op(
            self, op: AbstractOp, arguments: Dict[str, int],
            dbg_i: DebugInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, op, arguments, dbg_i=dbg_i, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_add_i(
            self, lhs: int, rhs: int,
            dbg_i: DebugInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.ADD_I(),
                           {"lhs": lhs, "rhs": rhs}, dbg_i=dbg_i, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_add_f(
            self, lhs: int, rhs: int,
            dbg_i: DebugInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.ADD_F(),
                           {"lhs": lhs, "rhs": rhs}, dbg_i=dbg_i, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_mul_i(
            self, lhs: int, rhs: int,
            dbg_i: DebugInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.MUL_I(),
                           {"lhs": lhs, "rhs": rhs}, dbg_i=dbg_i, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_mul_f(
            self, lhs: int, rhs: int,
            dbg_i: DebugInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.MUL_F(),
                           {"lhs": lhs, "rhs": rhs}, dbg_i=dbg_i, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_sub_i(
            self, lhs: int, rhs: int,
            dbg_i: DebugInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.SUB_I(),
                           {"lhs": lhs, "rhs": rhs}, dbg_i=dbg_i, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_sub_f(
            self, lhs: int, rhs: int,
            dbg_i: DebugInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.SUB_F(),
                           {"lhs": lhs, "rhs": rhs}, dbg_i=dbg_i, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_div_i(
            self, lhs: int, rhs: int,
            dbg_i: DebugInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.DIV_I(),
                           {"lhs": lhs, "rhs": rhs}, dbg_i=dbg_i, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_div_f(
            self, lhs: int, rhs: int,
            dbg_i: DebugInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.DIV_F(),
                           {"lhs": lhs, "rhs": rhs}, dbg_i=dbg_i, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_add(
            self, lhs: int, rhs: int,
            dbg_i: DebugInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.ADD(),
                           {"lhs": lhs, "rhs": rhs}, dbg_i=dbg_i, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_sub(
            self, lhs: int, rhs: int,
            dbg_i: DebugInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.SUB(),
                           {"lhs": lhs, "rhs": rhs}, dbg_i=dbg_i, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_mul(
            self, lhs: int, rhs: int,
            dbg_i: DebugInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.MUL(),
                           {"lhs": lhs, "rhs": rhs}, dbg_i=dbg_i, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_div(
            self, lhs: int, rhs: int,
            dbg_i: DebugInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.DIV(),
                           {"lhs": lhs, "rhs": rhs}, dbg_i=dbg_i, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_logical_and(
            self, lhs: int, rhs: int,
            dbg_i: DebugInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.AND(),
                           {"lhs": lhs, "rhs": rhs}, dbg_i=dbg_i, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_logical_or(
            self, lhs: int, rhs: int,
            dbg_i: DebugInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.OR(),
                           {"lhs": lhs, "rhs": rhs}, dbg_i=dbg_i, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_logical_not(
            self, val: int,
            dbg_i: DebugInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.NOT(),
                           {"x": val}, dbg_i=dbg_i, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_not_equal(
            self, lhs: int, rhs: int,
            dbg_i: DebugInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.NE(),
                           {"lhs": lhs, "rhs": rhs}, dbg_i=dbg_i, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_not_equal_i(
            self, lhs: int, rhs: int,
            dbg_i: DebugInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.NE_I(),
                           {"lhs": lhs, "rhs": rhs}, dbg_i=dbg_i, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_not_equal_f(
            self, lhs: int, rhs: int,
            dbg_i: DebugInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.NE_F(),
                           {"lhs": lhs, "rhs": rhs}, dbg_i=dbg_i, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_equal(
            self, lhs: int, rhs: int,
            dbg_i: DebugInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.EQ(),
                           {"lhs": lhs, "rhs": rhs}, dbg_i=dbg_i, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_equal_i(
            self, lhs: int, rhs: int,
            dbg_i: DebugInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.EQ_I(),
                           {"lhs": lhs, "rhs": rhs}, dbg_i=dbg_i, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_equal_f(
            self, lhs: int, rhs: int,
            dbg_i: DebugInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.EQ_F(),
                           {"lhs": lhs, "rhs": rhs}, dbg_i=dbg_i, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_less_than(
            self, lhs: int, rhs: int,
            dbg_i: DebugInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.LT(),
                           {"lhs": lhs, "rhs": rhs}, dbg_i=dbg_i, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_less_than_i(
            self, lhs: int, rhs: int,
            dbg_i: DebugInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.LT_I(),
                           {"lhs": lhs, "rhs": rhs}, dbg_i=dbg_i, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_less_than_f(
            self, lhs: int, rhs: int,
            dbg_i: DebugInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.LT_F(),
                           {"lhs": lhs, "rhs": rhs}, dbg_i=dbg_i, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_less_than_or_equal(
            self, lhs: int, rhs: int,
            dbg_i: DebugInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.LTE(),
                           {"lhs": lhs, "rhs": rhs}, dbg_i=dbg_i, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_less_than_or_equal_i(
            self, lhs: int, rhs: int,
            dbg_i: DebugInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.LTE_I(),
                           {"lhs": lhs, "rhs": rhs}, dbg_i=dbg_i, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_less_than_or_equal_f(
            self, lhs: int, rhs: int,
            dbg_i: DebugInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.LTE_F(),
                           {"lhs": lhs, "rhs": rhs}, dbg_i=dbg_i, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_greater_than(
            self, lhs: int, rhs: int,
            dbg_i: DebugInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.GT(),
                           {"lhs": lhs, "rhs": rhs}, dbg_i=dbg_i, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_greater_than_i(
            self, lhs: int, rhs: int,
            dbg_i: DebugInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.GT_I(),
                           {"lhs": lhs, "rhs": rhs}, dbg_i=dbg_i, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_greater_than_f(
            self, lhs: int, rhs: int,
            dbg_i: DebugInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.GT_F(),
                           {"lhs": lhs, "rhs": rhs}, dbg_i=dbg_i, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_greater_than_or_equal(
            self, lhs: int, rhs: int,
            dbg_i: DebugInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.GTE(),
                           {"lhs": lhs, "rhs": rhs}, dbg_i=dbg_i, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_greater_than_or_equal_i(
            self, lhs: int, rhs: int,
            dbg_i: DebugInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.GTE_I(),
                           {"lhs": lhs, "rhs": rhs}, dbg_i=dbg_i, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_greater_than_or_equal_f(
            self, lhs: int, rhs: int,
            dbg_i: DebugInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.GTE_F(),
                           {"lhs": lhs, "rhs": rhs}, dbg_i=dbg_i, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_bool_cast(
            self, value: int,
            dbg_i: DebugInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.BOOL_CAST(),
                           {"x": value}, dbg_i=dbg_i, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_float_cast(
            self, value: int,
            dbg_i: DebugInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.FLOAT_CAST(),
                           {"x": value}, dbg_i=dbg_i, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_int_cast(
            self, value: int,
            dbg_i: DebugInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.INT_CAST(),
                           {"x": value}, dbg_i=dbg_i, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_select(
            self, condition: int, lhs: int, rhs: int,
            dbg_i: DebugInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.SELECT(),
                           {"cond": condition, "tv": lhs, "fv": rhs}, dbg_i=dbg_i, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_constant(
            self, value: int,
            dbg_i: DebugInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.CONSTANT(value),
                           {}, dbg_i=dbg_i, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_constant_float(
            self, value: float,
            dbg_i: DebugInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.CONSTANT_FLOAT(value),
                           {}, dbg_i=dbg_i, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_constant_none(
            self,
            dbg_i: DebugInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.CONSTANT_NONE(),
                           {}, dbg_i=dbg_i, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_constant_datatype(
            self, dt: DTDescriptor,
            dbg_i: DebugInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.CONSTANT_CLASS(dt),
                           {}, dbg_i=dbg_i, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_constant_cast(
            self, value: int,
            dbg_i: DebugInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.CONSTANT_CAST(),
                           {"x": value}, dbg_i=dbg_i, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_slicing(
            self, value: int, slicing_params: List[Tuple[int, ...]],
            dbg_i: DebugInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.SLICE(slicing_params),
                           {"self": value}, dbg_i=dbg_i, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_slicing_assign(
            self, slicing_params_list: List[List[Tuple[int, ...]]], orig_value: int, value: int,
            dbg_i: DebugInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.ASSIGN_SLICE(slicing_params_list),
                           {"self": orig_value, "value": value}, dbg_i=dbg_i, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_square_brackets(
            self, values: List[int],
            dbg_i: DebugInfo = None, annotation: Annotation | None = None
    ) -> int:
        operator = Operators.NoCls.SQUARE_BRACKETS()
        params = operator.params_parse(dbg_i, values, {})
        stmt = IRStatement(self._next_id, operator, params, dbg_i=dbg_i, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_parenthesis(
            self, values: List[int],
            dbg_i: DebugInfo = None, annotation: Annotation | None = None
    ) -> int:
        operator = Operators.NoCls.PARENTHESIS()
        params = operator.params_parse(dbg_i, values, {})
        stmt = IRStatement(self._next_id, operator, params, dbg_i=dbg_i, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_assert(
            self, value: int,
            dbg_i: DebugInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.ASSERT(),
                           {"test": value}, dbg_i=dbg_i, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_input(
            self, input_id: int, dt: DTDescriptor, public: bool,
            dbg_i: DebugInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.INPUT(input_id, dt, public),
                           {}, dbg_i=dbg_i, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        self._next_id += 1
        return self._next_id - 1

    def create_read_integer(
            self, major: int, minor: int,
            dbg_i: DebugInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.READ_INTEGER(major, minor),
                           {}, dbg_i=dbg_i, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_read_float(
            self, major: int, minor: int,
            dbg_i: DebugInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.READ_FLOAT(major, minor),
                           {}, dbg_i=dbg_i, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_similar(
            self, ir_stmt: IRStatement, args: Dict[str, int],
            dbg_i: DebugInfo = None
    ) -> int:
        new_stmt = IRStatement(
            self._next_id,
            operator=ir_stmt.operator,
            arguments=args,
            annotation=None,
            dbg_i=dbg_i
        )
        new_stmt = self._do_ir_inference(new_stmt)
        _check_annotation_and_raise(new_stmt.annotation, ir_stmt.annotation)
        if new_stmt.annotation is None:
            new_stmt.annotation = ir_stmt.annotation
        self.stmts.append(new_stmt)
        self._next_id += 1
        return self._next_id - 1

    def _do_ir_inference(self, stmt: IRStatement) -> IRStatement:
        if self.ir_ctx is None:
            return stmt
        descriptors = {name: self.ir_ctx.get_inference_descriptor(ptr) for name, ptr in stmt.arguments.items()}
        dt_descriptor = stmt.operator.type_check(stmt.dbg_i, descriptors)
        inference_descriptor = stmt.operator.static_infer(stmt.dbg_i, descriptors)
        self.ir_ctx.set_inference_descriptor(stmt.stmt_id, inference_descriptor)
        # TODO
        # if dt_descriptor is not None:
        #     stmt.annotation = Annotation(dt_descriptor.typename)
        return stmt

    def export_ir_graph(self) -> IRGraph:
        return IRGraph(self.stmts, IRGraphMetadata(annotated=self.ir_ctx is not None))
