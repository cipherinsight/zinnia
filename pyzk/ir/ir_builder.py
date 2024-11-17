from typing import Tuple, List

from .ir_ctx import IRContext
from .ir_graph import IRGraph, IRGraphMetadata
from .ir_stmt import IRStatement
from ..inference.ir_inference import IRInferenceDescriptor, IRInference
from ..util.annotation import Annotation
from ..util.op_name import OpName
from ..util.source_pos_info import SourcePosInfo


def _check_annotation_and_raise(lhs: Annotation | None, rhs: Annotation | None):
    if lhs is None or rhs is None:
        return
    

class IRBuilder:
    def __init__(self, ir_ctx: IRContext | None = None) -> None:
        self.stmts = []
        self._next_id = len(self.stmts)
        self.ir_ctx = ir_ctx

    def create_op(
            self, op_name: str, op_args: List[int], constant_args: List[int] | None = None,
            source_pos_info: SourcePosInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, op_name, op_args, constant_args=constant_args, source_pos_info=source_pos_info)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_add(
            self, lhs: int, rhs: int,
            source_pos_info: SourcePosInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, OpName.Binary.ADD, [lhs, rhs], source_pos_info=source_pos_info)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_sub(
            self, lhs: int, rhs: int,
            source_pos_info: SourcePosInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, OpName.Binary.SUB, [lhs, rhs], source_pos_info=source_pos_info)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_mul(
            self, lhs: int, rhs: int,
            source_pos_info: SourcePosInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, OpName.Binary.MUL, [lhs, rhs], source_pos_info=source_pos_info)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_div(
            self, lhs: int, rhs: int,
            source_pos_info: SourcePosInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, OpName.Binary.DIV, [lhs, rhs], source_pos_info=source_pos_info)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_logical_and(
            self, lhs: int, rhs: int,
            source_pos_info: SourcePosInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, OpName.Binary.AND, [lhs, rhs], source_pos_info=source_pos_info)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_logical_or(
            self, lhs: int, rhs: int,
            source_pos_info: SourcePosInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, OpName.Binary.OR, [lhs, rhs], source_pos_info=source_pos_info)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_logical_not(
            self, val: int,
            source_pos_info: SourcePosInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, OpName.Unary.NOT, [val], source_pos_info=source_pos_info)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_is_true(
            self, value: int,
            source_pos_info: SourcePosInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, OpName.Unary.IS_TRUE, [value], source_pos_info=source_pos_info)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_is_false(
            self, value: int,
            source_pos_info: SourcePosInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, OpName.Unary.IS_FALSE, [value], source_pos_info=source_pos_info)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_constant(
            self, value: int,
            source_pos_info: SourcePosInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, OpName.Special.CONSTANT, [], constant_value=value, source_pos_info=source_pos_info)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_slicing(
            self, value: int, slicing: List[int | Tuple[int, int]],
            source_pos_info: SourcePosInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, OpName.Special.SLICING, [value], slicing_args=slicing, source_pos_info=source_pos_info)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_slicing_assign(
            self, slicing: List[List[int | Tuple[int, int]]], orig_value: int, value: int,
            source_pos_info: SourcePosInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, OpName.Special.SLICING_ASSIGN, [orig_value, value], slicing_assign_args=slicing, source_pos_info=source_pos_info)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_concat(
            self, values: List[int],
            source_pos_info: SourcePosInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, OpName.Special.CONCAT, values, source_pos_info=source_pos_info)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_new_list(
            self, values: List[int],
            source_pos_info: SourcePosInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, OpName.Special.NEW_LIST, values, source_pos_info=source_pos_info)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_assert(
            self, value: int,
            source_pos_info: SourcePosInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, OpName.Special.ASSERT, [value], source_pos_info=source_pos_info)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_input(
            self, input_id: int,
            source_pos_info: SourcePosInfo = None, annotation: Annotation | None = None
    ) -> int:
        assert annotation is not None, "You must specify an annotation for `input` operator."
        stmt = IRStatement(self._next_id, OpName.Special.INPUT, [], constant_args=[input_id], source_pos_info=source_pos_info)
        stmt.annotation = annotation
        if self.ir_ctx is not None:
            self.ir_ctx.set_inference_descriptor(stmt.stmt_id, IRInferenceDescriptor.new(
                annotation.typename, annotation.shape, annotation.public, None
            ))
        self.stmts.append(stmt)
        self._next_id += 1
        return self._next_id - 1

    def create_read_int(
            self, input_id: int, idx: int,
            source_pos_info: SourcePosInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, OpName.Special.READ_INT, [], constant_args=[input_id, idx], source_pos_info=source_pos_info)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_expose_public(
            self, val: int,
            source_pos_info: SourcePosInfo = None, annotation: Annotation | None = None
    ) -> int:
        stmt = IRStatement(self._next_id, OpName.Special.EXPOSE_PUBLIC, [val], source_pos_info=source_pos_info)
        self.stmts.append(self._do_ir_inference(stmt))
        _check_annotation_and_raise(stmt.annotation, annotation)
        self._next_id += 1
        return self._next_id - 1

    def create_similar(
            self, ir_stmt: IRStatement, args: List[int],
            source_pos_info: SourcePosInfo = None
    ) -> int:
        new_stmt = IRStatement(
            self._next_id,
            op_name=ir_stmt.op,
            op_args=args,
            constant_value=ir_stmt.constant_value,
            slicing_args=ir_stmt.slicing_args,
            slicing_assign_args=ir_stmt.slicing_assign_args,
            constant_args=ir_stmt.constant_args,
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
        args = stmt.args
        descriptors = [self.ir_ctx.get_inference_descriptor(ptr) for ptr in stmt.args]
        result_descriptor = IRInference.do_ir_inference(stmt.op, descriptors, stmt.constant_args, stmt.slicing_args, stmt.slicing_assign_args, stmt.source_pos_info, stmt.constant_value)
        self.ir_ctx.set_inference_descriptor(stmt.stmt_id, result_descriptor)
        if result_descriptor is not None:
            stmt.annotation = Annotation(result_descriptor.typename, result_descriptor.get_shape(), result_descriptor.public)
        return stmt        

    def export_ir_graph(self) -> IRGraph:
        return IRGraph(self.stmts, IRGraphMetadata(annotated=self.ir_ctx is not None))
