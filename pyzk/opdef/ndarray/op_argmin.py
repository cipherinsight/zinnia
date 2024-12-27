from typing import Any, Tuple

from pyzk.internal.dt_descriptor import DTDescriptor, IntegerDTDescriptor, FloatDTDescriptor
from pyzk.opdef.ndarray.abstract_aggregator import AbstractAggregator


class NDArray_ArgMinOp(AbstractAggregator):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "NDArray::argmin"

    @classmethod
    def get_name(cls) -> str:
        return "NDArray::argmin"

    def get_result_dtype(self, element_dt: DTDescriptor):
        return IntegerDTDescriptor()

    def aggregator_func(self, lhs: Any, lhs_i: int, rhs: Any, rhs_i: int, dt: DTDescriptor) -> Tuple[Any, int | None]:
        return min(lhs, rhs) if lhs is not None and rhs is not None else None, (rhs_i if rhs < lhs else lhs_i) if lhs is not None and rhs is not None else None

    def initial_func(self, dt: DTDescriptor, first_ele: Any) -> Tuple[Any, int | None]:
        return first_ele, 0

    def parsing_func(self, lhs: Any, rhs: Any) -> Any:
        return rhs

    def aggregator_build_ir(self, ir_builder, lhs: int, lhs_i: int, rhs: int, rhs_i: int, dt: DTDescriptor) -> Tuple[int, int | None]:
        if isinstance(dt, IntegerDTDescriptor):
            cond = ir_builder.create_less_than_i(lhs, rhs)
            not_cond = ir_builder.create_logical_not(cond)
            candidate = ir_builder.create_add_i(
                ir_builder.create_mul_i(not_cond, rhs),
                ir_builder.create_mul_i(cond, lhs)
            )
            candidate_i = ir_builder.create_add_i(
                ir_builder.create_mul_i(not_cond, ir_builder.create_constant(rhs_i)),
                ir_builder.create_mul_i(cond, lhs_i)
            )
            return candidate, candidate_i
        elif isinstance(dt, FloatDTDescriptor):
            cond = ir_builder.create_less_than_f(lhs, rhs)
            not_cond = ir_builder.create_logical_not(cond)
            candidate = ir_builder.create_add_f(
                ir_builder.create_mul_f(ir_builder.create_float_cast(not_cond), rhs),
                ir_builder.create_mul_f(ir_builder.create_float_cast(cond), lhs)
            )
            candidate_i = ir_builder.create_add_i(
                ir_builder.create_mul_i(not_cond, ir_builder.create_constant(rhs_i)),
                ir_builder.create_mul_i(cond, lhs_i)
            )
            return candidate, candidate_i
        raise NotImplementedError()

    def initial_build_ir(self, ir_builder, dt: DTDescriptor, first_ele: int) -> Tuple[int, int | None]:
        return first_ele, ir_builder.create_constant(0)

    def parsing_select_ir(self, ir_builder, lhs: int, rhs: int) -> int:
        return rhs
