from typing import Any, Tuple

from pyzk.internal.dt_descriptor import DTDescriptor, IntegerDTDescriptor, FloatDTDescriptor
from pyzk.opdef.ndarray.abstract_aggregator import AbstractAggregator


class NDArray_MinOp(AbstractAggregator):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "NDArray::min"

    @classmethod
    def get_name(cls) -> str:
        return "NDArray::min"

    def aggregator_func(self, lhs: Any, lhs_i: int, rhs: Any, rhs_i: int, dt: DTDescriptor) -> Tuple[Any, int | None]:
        if isinstance(dt, IntegerDTDescriptor):
            return min(lhs, rhs) if lhs is not None and rhs is not None else None, None
        return None, None

    def initial_func(self, dt: DTDescriptor, first_ele: Any) -> Tuple[Any, int | None]:
        return first_ele, None

    def aggregator_build_ir(self, ir_builder, lhs: int, lhs_i: int, rhs: int, rhs_i: int, dt: DTDescriptor) -> Tuple[int, int | None]:
        if isinstance(dt, IntegerDTDescriptor):
            cond = ir_builder.create_less_than_i(lhs, rhs)
            not_cond = ir_builder.create_logical_not(cond)
            candidate = ir_builder.create_add_i(
                ir_builder.create_mul_i(not_cond, rhs),
                ir_builder.create_mul_i(cond, lhs)
            )
            return candidate, None
        elif isinstance(dt, FloatDTDescriptor):
            cond = ir_builder.create_less_than_f(lhs, rhs)
            not_cond = ir_builder.create_logical_not(cond)
            candidate = ir_builder.create_add_f(
                ir_builder.create_mul_f(ir_builder.create_float_cast(not_cond), rhs),
                ir_builder.create_mul_f(ir_builder.create_float_cast(cond), lhs)
            )
            return candidate, None
        raise NotImplementedError()

    def initial_build_ir(self, ir_builder, dt: DTDescriptor, first_ele: int) -> Tuple[int, int | None]:
        return first_ele, None
