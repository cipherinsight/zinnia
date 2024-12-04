from typing import Tuple, List, Dict

from pyzk.ir.ir_ctx import IRContext
from pyzk.ir.ir_graph import IRGraph, IRGraphMetadata
from pyzk.ir.ir_stmt import IRStatement
from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.util.annotation import Annotation
from pyzk.util.dt_descriptor import DTDescriptor
from pyzk.opdef.operator_factory import Operators
from pyzk.util.source_pos_info import SourcePosInfo


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
            source_pos_info: SourcePosInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, op, arguments, source_pos_info=source_pos_info, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_add(
            self, lhs: int, rhs: int,
            source_pos_info: SourcePosInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.ADD(),
                           {"lhs": lhs, "rhs": rhs}, source_pos_info=source_pos_info, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_sub(
            self, lhs: int, rhs: int,
            source_pos_info: SourcePosInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.SUB(),
                           {"lhs": lhs, "rhs": rhs}, source_pos_info=source_pos_info, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_mul(
            self, lhs: int, rhs: int,
            source_pos_info: SourcePosInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.MUL(),
                           {"lhs": lhs, "rhs": rhs}, source_pos_info=source_pos_info, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_div(
            self, lhs: int, rhs: int,
            source_pos_info: SourcePosInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.DIV(),
                           {"lhs": lhs, "rhs": rhs}, source_pos_info=source_pos_info, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_logical_and(
            self, lhs: int, rhs: int,
            source_pos_info: SourcePosInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.AND(),
                           {"lhs": lhs, "rhs": rhs}, source_pos_info=source_pos_info, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_logical_or(
            self, lhs: int, rhs: int,
            source_pos_info: SourcePosInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.OR(),
                           {"lhs": lhs, "rhs": rhs}, source_pos_info=source_pos_info, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_logical_not(
            self, val: int,
            source_pos_info: SourcePosInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.NOT(),
                           {"x": val}, source_pos_info=source_pos_info, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_not_equal(
            self, lhs: int, rhs: int,
            source_pos_info: SourcePosInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.NE(),
                           {"lhs": lhs, "rhs": rhs}, source_pos_info=source_pos_info, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_equal(
            self, lhs: int, rhs: int,
            source_pos_info: SourcePosInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.EQ(),
                           {"lhs": lhs, "rhs": rhs}, source_pos_info=source_pos_info, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_less_than(
            self, lhs: int, rhs: int,
            source_pos_info: SourcePosInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.LT(),
                           {"lhs": lhs, "rhs": rhs}, source_pos_info=source_pos_info, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_less_than_or_equal(
            self, lhs: int, rhs: int,
            source_pos_info: SourcePosInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.LTE(),
                           {"lhs": lhs, "rhs": rhs}, source_pos_info=source_pos_info, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_greater_than(
            self, lhs: int, rhs: int,
            source_pos_info: SourcePosInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.GT(),
                           {"lhs": lhs, "rhs": rhs}, source_pos_info=source_pos_info, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_greater_than_or_equal(
            self, lhs: int, rhs: int,
            source_pos_info: SourcePosInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.GTE(),
                           {"lhs": lhs, "rhs": rhs}, source_pos_info=source_pos_info, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_bool_cast(
            self, value: int,
            source_pos_info: SourcePosInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.BOOL_CAST(),
                           {"x": value}, source_pos_info=source_pos_info, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_constant(
            self, value: int,
            source_pos_info: SourcePosInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.CONSTANT(value),
                           {}, source_pos_info=source_pos_info, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_constant_cast(
            self, value: int,
            source_pos_info: SourcePosInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.CONSTANT_CAST(),
                           {"x": value}, source_pos_info=source_pos_info, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_slicing(
            self, value: int, slicing_params: List[Tuple[int, ...]],
            source_pos_info: SourcePosInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.SLICE(slicing_params),
                           {"self": value}, source_pos_info=source_pos_info, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_slicing_assign(
            self, slicing_params_list: List[List[Tuple[int, ...]]], orig_value: int, value: int,
            source_pos_info: SourcePosInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.ASSIGN_SLICE(slicing_params_list),
                           {"self": orig_value, "value": value}, source_pos_info=source_pos_info, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_square_brackets(
            self, values: List[int],
            source_pos_info: SourcePosInfo = None, annotation: Annotation | None = None
    ) -> int:
        operator = Operators.instantiate_operator("square_brackets", None)
        params = operator.params_parse(source_pos_info, values, {})
        stmt = IRStatement(self._next_id, operator, params, source_pos_info=source_pos_info, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_parenthesis(
            self, values: List[int],
            source_pos_info: SourcePosInfo = None, annotation: Annotation | None = None
    ) -> int:
        operator = Operators.NoCls.PARENTHESIS()
        params = operator.params_parse(source_pos_info, values, {})
        stmt = IRStatement(self._next_id, operator, params, source_pos_info=source_pos_info, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_assert(
            self, value: int,
            source_pos_info: SourcePosInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.ASSERT(),
                           {"test": value}, source_pos_info=source_pos_info, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_input(
            self, input_id: int, dt: DTDescriptor, public: bool,
            source_pos_info: SourcePosInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.INPUT(input_id, dt, public),
                           {}, source_pos_info=source_pos_info, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        self._next_id += 1
        return self._next_id - 1

    def create_read_number(
            self, major: int, minor: int,
            source_pos_info: SourcePosInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, Operators.NoCls.READ_NUMBER(major, minor),
                           {}, source_pos_info=source_pos_info, annotation=annotation)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_similar(
            self, ir_stmt: IRStatement, args: Dict[str, int],
            source_pos_info: SourcePosInfo = None
    ) -> int:
        new_stmt = IRStatement(
            self._next_id,
            operator=ir_stmt.operator,
            arguments=args,
            annotation=None,
            source_pos_info=source_pos_info
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
        dt_descriptor = stmt.operator.type_check(stmt.source_pos_info, descriptors)
        inference_descriptor = stmt.operator.static_infer(stmt.source_pos_info, descriptors)
        self.ir_ctx.set_inference_descriptor(stmt.stmt_id, inference_descriptor)
        # TODO
        # if dt_descriptor is not None:
        #     stmt.annotation = Annotation(dt_descriptor.typename)
        return stmt

    def export_ir_graph(self) -> IRGraph:
        return IRGraph(self.stmts, IRGraphMetadata(annotated=self.ir_ctx is not None))
