from typing import Callable, Any, Dict, Optional

from pyzk.opdef.nocls.abstract_arithemetic import AbstractArithemetic
from pyzk.internal.dt_descriptor import DTDescriptor, TupleDTDescriptor
from pyzk.internal.flatten_descriptor import FlattenDescriptor, TupleFlattenDescriptor
from pyzk.internal.inference_descriptor import InferenceDescriptor, TupleInferenceDescriptor
from pyzk.debug.dbg_info import DebugInfo


class AddOp(AbstractArithemetic):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "add"

    @classmethod
    def get_name(cls) -> str:
        return "add"

    def get_inference_op_lambda(self) -> Callable[[Any, Any], Any]:
        return lambda x, y: x + y if x is not None and y is not None else None

    def get_flatten_op_lambda(self, ir_builder) -> Callable[[int, int], int]:
        return lambda x, y: ir_builder.create_add(x, y)

    def type_check(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        lhs, rhs = kwargs["lhs"].type(), kwargs["rhs"].type()
        if isinstance(lhs, TupleDTDescriptor) and isinstance(rhs, TupleDTDescriptor):
            return TupleDTDescriptor(lhs.length + rhs.length)
        return super().type_check(dbg_i, kwargs)

    def static_infer(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        lhs, rhs = kwargs["lhs"], kwargs["rhs"]
        if isinstance(lhs, TupleInferenceDescriptor) and isinstance(rhs, TupleInferenceDescriptor):
            return TupleInferenceDescriptor(lhs.length() + rhs.length(), lhs.get() + rhs.get())
        return super().static_infer(dbg_i, kwargs)

    def ir_flatten(self, ir_builder, kwargs: Dict[str, FlattenDescriptor]) -> FlattenDescriptor:
        lhs, rhs = kwargs["lhs"], kwargs["rhs"]
        if isinstance(lhs, TupleFlattenDescriptor) and isinstance(rhs, TupleFlattenDescriptor):
            return TupleFlattenDescriptor(lhs.length() + rhs.length(), lhs.ptr() + rhs.ptr())
        return super().ir_flatten(ir_builder, kwargs)
