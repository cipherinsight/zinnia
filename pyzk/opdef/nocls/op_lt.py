from typing import Callable, Any

from pyzk.internal.dt_descriptor import DTDescriptor, IntegerDTDescriptor, FloatDTDescriptor
from pyzk.opdef.nocls.abstract_compare import AbstractCompare


class LessThanOp(AbstractCompare):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "lt"

    @classmethod
    def get_name(cls) -> str:
        return "lt"

    def get_inference_op_lambda(self, lhs_dt: DTDescriptor, rhs_dt: DTDescriptor) -> Callable[[Any, Any], Any]:
        return lambda x, y: (1 if x < y else 0) if x is not None and y is not None else None

    def get_flatten_op_lambda(self, ir_builder, lhs_dt: DTDescriptor, rhs_dt: DTDescriptor) -> Callable[[int, int], int]:
        if isinstance(lhs_dt, IntegerDTDescriptor) and isinstance(rhs_dt, IntegerDTDescriptor):
            return lambda x, y: ir_builder.create_less_than_i(x, y)
        elif isinstance(lhs_dt, FloatDTDescriptor) and isinstance(rhs_dt, IntegerDTDescriptor):
            return lambda x, y: ir_builder.create_less_than_f(x, ir_builder.create_float_cast(y))
        elif isinstance(lhs_dt, IntegerDTDescriptor) and isinstance(rhs_dt, FloatDTDescriptor):
            return lambda x, y: ir_builder.create_less_than_f(ir_builder.create_float_cast(x), y)
        elif isinstance(lhs_dt, FloatDTDescriptor) and isinstance(rhs_dt, FloatDTDescriptor):
            return lambda x, y: ir_builder.create_less_than_f(x, y)
        raise NotImplementedError()
