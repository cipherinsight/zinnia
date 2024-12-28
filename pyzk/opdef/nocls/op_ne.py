from typing import Callable, Any, Optional, Dict

from pyzk.debug.dbg_info import DebugInfo
from pyzk.debug.exception import TypeInferenceError
from pyzk.internal.dt_descriptor import IntegerDTDescriptor, FloatDTDescriptor, DTDescriptor, TupleDTDescriptor
from pyzk.internal.flatten_descriptor import FlattenDescriptor, IntegerFlattenDescriptor
from pyzk.internal.inference_descriptor import InferenceDescriptor, IntegerInferenceDescriptor
from pyzk.opdef.nocls.abstract_compare import AbstractCompare


class NotEqualOp(AbstractCompare):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "ne"

    @classmethod
    def get_name(cls) -> str:
        return "ne"

    def get_inference_op_lambda(self, lhs_dt: DTDescriptor, rhs_dt: DTDescriptor) -> Callable[[Any, Any], Any]:
        return lambda x, y: (1 if x != y else 0) if x is not None and y is not None else None

    def get_flatten_op_lambda(self, ir_builder, lhs_dt: DTDescriptor, rhs_dt: DTDescriptor) -> Callable[[int, int], int]:
        if isinstance(lhs_dt, IntegerDTDescriptor) and isinstance(rhs_dt, IntegerDTDescriptor):
            return lambda x, y: ir_builder.create_not_equal_i(x, y)
        elif isinstance(lhs_dt, FloatDTDescriptor) and isinstance(rhs_dt, IntegerDTDescriptor):
            return lambda x, y: ir_builder.create_not_equal_f(x, ir_builder.create_float_cast(y))
        elif isinstance(lhs_dt, IntegerDTDescriptor) and isinstance(rhs_dt, FloatDTDescriptor):
            return lambda x, y: ir_builder.create_not_equal_f(ir_builder.create_float_cast(x), y)
        elif isinstance(lhs_dt, FloatDTDescriptor) and isinstance(rhs_dt, FloatDTDescriptor):
            return lambda x, y: ir_builder.create_not_equal_f(x, y)
        raise NotImplementedError()

    def type_check(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        lhs, rhs = kwargs["lhs"].type(), kwargs["rhs"].type()
        if isinstance(lhs, TupleDTDescriptor) and isinstance(rhs, TupleDTDescriptor):
            if lhs.length != rhs.length:
                raise TypeInferenceError(dbg_i, f'Invalid binary operator `{self.get_name()}` on tuple operands {lhs} and {rhs}, as their lengths must be equal')
            return IntegerDTDescriptor()
        elif isinstance(lhs, TupleDTDescriptor) or isinstance(rhs, TupleDTDescriptor):
            raise TypeInferenceError(dbg_i, f'Invalid binary operator `{self.get_name()}` on tuple operands {lhs} and {rhs}, as both operands must be tuples')
        return super().type_check(dbg_i, kwargs)

    def static_infer(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        lhs, rhs = kwargs["lhs"], kwargs["rhs"]
        if isinstance(lhs.type(), TupleDTDescriptor) and isinstance(rhs.type(), TupleDTDescriptor):
            return IntegerInferenceDescriptor(1 if lhs.get() != rhs.get() else 0)
        return super().static_infer(dbg_i, kwargs)

    def ir_flatten(self, ir_builder, kwargs: Dict[str, FlattenDescriptor]) -> FlattenDescriptor:
        lhs, rhs = kwargs["lhs"], kwargs["rhs"]
        if isinstance(lhs.type(), TupleDTDescriptor) and isinstance(rhs.type(), TupleDTDescriptor):
            return IntegerFlattenDescriptor(ir_builder.create_constant(1 if lhs.val() != rhs.val() else 0))
        return super().ir_flatten(ir_builder, kwargs)
