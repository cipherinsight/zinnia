from typing import Callable, Any

from pyzk.internal.dt_descriptor import IntegerDTDescriptor, DTDescriptor, FloatDTDescriptor
from pyzk.opdef.nocls.abstract_arithemetic import AbstractArithemetic


class FloorDivideOp(AbstractArithemetic):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "floor_divide"

    @classmethod
    def get_name(cls) -> str:
        return "floor_divide"

    def get_inference_op_lambda(self, lhs_dt: DTDescriptor, rhs_dt: DTDescriptor) -> Callable[[Any, Any], Any]:
        if isinstance(lhs_dt, IntegerDTDescriptor) and isinstance(rhs_dt, IntegerDTDescriptor):
            return lambda x, y: x // y if x is not None and y is not None else None
        else:
            return lambda x, y: None

    def get_flatten_op_lambda(self, ir_builder, lhs_dt: DTDescriptor, rhs_dt: DTDescriptor) -> Callable[[int, int], int]:
        if isinstance(lhs_dt, IntegerDTDescriptor) and isinstance(rhs_dt, IntegerDTDescriptor):
            return lambda x, y: ir_builder.create_div_i(ir_builder.create_sub_i(x, ir_builder.create_mod_i(x, y)), y)
        elif isinstance(lhs_dt, FloatDTDescriptor) and isinstance(rhs_dt, IntegerDTDescriptor):
            return lambda x, y: ir_builder.create_div_f(ir_builder.create_sub_f(x, ir_builder.create_mod_f(x, ir_builder.create_float_cast(y))), ir_builder.create_float_cast(y))
        elif isinstance(lhs_dt, IntegerDTDescriptor) and isinstance(rhs_dt, FloatDTDescriptor):
            return lambda x, y: ir_builder.create_div_f(ir_builder.create_sub_f(x, ir_builder.create_mod_f(ir_builder.create_float_cast(x), y)), y)
        elif isinstance(lhs_dt, FloatDTDescriptor) and isinstance(rhs_dt, FloatDTDescriptor):
            return lambda x, y: ir_builder.create_div_f(ir_builder.create_sub_f(x, ir_builder.create_mod_f(x, y)), y)
        raise NotImplementedError()
