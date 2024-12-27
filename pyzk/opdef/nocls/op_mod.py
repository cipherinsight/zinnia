from typing import Callable, Any

from pyzk.internal.dt_descriptor import DTDescriptor, IntegerDTDescriptor, FloatDTDescriptor
from pyzk.opdef.nocls.abstract_arithemetic import AbstractArithemetic


class ModOp(AbstractArithemetic):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "mod"

    @classmethod
    def get_name(cls) -> str:
        return "mod"

    def get_inference_op_lambda(self, lhs_dt: DTDescriptor, rhs_dt: DTDescriptor) -> Callable[[Any, Any], Any]:
        if isinstance(lhs_dt, IntegerDTDescriptor) and isinstance(rhs_dt, IntegerDTDescriptor):
            return lambda x, y: x % y if x is not None and y is not None else None
        else:
            return lambda x, y: None

    def get_flatten_op_lambda(self, ir_builder, lhs_dt: DTDescriptor, rhs_dt: DTDescriptor) -> Callable[[int, int], int]:
        if isinstance(lhs_dt, IntegerDTDescriptor) and isinstance(rhs_dt, IntegerDTDescriptor):
            return lambda x, y: ir_builder.create_mod_i(x, y)
        elif isinstance(lhs_dt, FloatDTDescriptor) and isinstance(rhs_dt, IntegerDTDescriptor):
            return lambda x, y: ir_builder.create_mod_f(x, ir_builder.create_float_cast(y))
        elif isinstance(lhs_dt, IntegerDTDescriptor) and isinstance(rhs_dt, FloatDTDescriptor):
            return lambda x, y: ir_builder.create_mod_f(ir_builder.create_float_cast(x), y)
        elif isinstance(lhs_dt, FloatDTDescriptor) and isinstance(rhs_dt, FloatDTDescriptor):
            return lambda x, y: ir_builder.create_mod_f(x, y)
        raise NotImplementedError()
