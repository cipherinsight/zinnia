from typing import Callable, Any

from pyzk.internal.dt_descriptor import IntegerDTDescriptor, DTDescriptor, FloatDTDescriptor
from pyzk.opdef.nocls.abstract_arithemetic import AbstractArithemetic


class MinimumOp(AbstractArithemetic):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "minimum"

    @classmethod
    def get_name(cls) -> str:
        return "minimum"

    def get_inference_op_lambda(self, lhs_dt: DTDescriptor, rhs_dt: DTDescriptor) -> Callable[[Any, Any], Any]:
        if isinstance(lhs_dt, IntegerDTDescriptor) and isinstance(rhs_dt, IntegerDTDescriptor):
            return lambda x, y: min(x, y) if x is not None and y is not None else None
        return lambda x, y: float(min(x, y)) if x is not None and y is not None else None

    def get_flatten_op_lambda(self, ir_builder, lhs_dt: DTDescriptor, rhs_dt: DTDescriptor) -> Callable[[int, int], int]:
        if isinstance(lhs_dt, IntegerDTDescriptor) and isinstance(rhs_dt, IntegerDTDescriptor):
            def _inner(lhs: int, rhs: int) -> int:
                cond = ir_builder.create_less_than_i(lhs, rhs)
                not_cond = ir_builder.create_logical_not(cond)
                candidate = ir_builder.create_add_i(
                    ir_builder.create_mul_i(not_cond, rhs),
                    ir_builder.create_mul_i(cond, lhs)
                )
                return candidate
            return _inner
        elif isinstance(lhs_dt, FloatDTDescriptor) and isinstance(rhs_dt, IntegerDTDescriptor):
            def _inner(lhs: int, rhs: int) -> int:
                rhs = ir_builder.create_float_cast(rhs)
                cond = ir_builder.create_less_than_f(lhs, rhs)
                not_cond = ir_builder.create_logical_not(cond)
                candidate = ir_builder.create_add_f(
                    ir_builder.create_mul_f(ir_builder.create_float_cast(not_cond), rhs),
                    ir_builder.create_mul_f(ir_builder.create_float_cast(cond), lhs)
                )
                return candidate
            return _inner
        elif isinstance(lhs_dt, IntegerDTDescriptor) and isinstance(rhs_dt, FloatDTDescriptor):
            def _inner(lhs: int, rhs: int) -> int:
                lhs = ir_builder.create_float_cast(lhs)
                cond = ir_builder.create_less_than_f(lhs, rhs)
                not_cond = ir_builder.create_logical_not(cond)
                candidate = ir_builder.create_add_f(
                    ir_builder.create_mul_f(ir_builder.create_float_cast(not_cond), rhs),
                    ir_builder.create_mul_f(ir_builder.create_float_cast(cond), lhs)
                )
                return candidate
            return _inner
        elif isinstance(lhs_dt, FloatDTDescriptor) and isinstance(rhs_dt, FloatDTDescriptor):
            def _inner(lhs: int, rhs: int) -> int:
                cond = ir_builder.create_less_than_f(lhs, rhs)
                not_cond = ir_builder.create_logical_not(cond)
                candidate = ir_builder.create_add_f(
                    ir_builder.create_mul_f(ir_builder.create_float_cast(not_cond), rhs),
                    ir_builder.create_mul_f(ir_builder.create_float_cast(cond), lhs)
                )
                return candidate
            return _inner
        raise NotImplementedError()
