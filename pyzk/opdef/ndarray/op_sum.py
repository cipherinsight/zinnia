from typing import Any, Tuple

from pyzk.internal.dt_descriptor import DTDescriptor, IntegerDTDescriptor
from pyzk.opdef.ndarray.abstract_aggregator import AbstractAggregator


class NDArray_SumOp(AbstractAggregator):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "NDArray::sum"

    @classmethod
    def get_name(cls) -> str:
        return "NDArray::sum"

    def aggregator_func(self, lhs: Any, lhs_i: int, rhs: Any, rhs_i: int, dt: DTDescriptor) -> Tuple[Any, int | None]:
        if isinstance(dt, IntegerDTDescriptor):
            return (lhs + rhs) if lhs is not None and rhs is not None else None, None
        return None, None

    def initial_func(self, dt: DTDescriptor, first_ele: Any) -> Tuple[Any, int | None]:
        if isinstance(dt, IntegerDTDescriptor):
            return 0, None
        return 0.0, None

    def aggregator_build_ir(self, ir_builder, lhs: int, lhs_i: int, rhs: int, rhs_i: int, dt: DTDescriptor) -> Tuple[int, int | None]:
        if isinstance(dt, IntegerDTDescriptor):
            return ir_builder.create_add_i(lhs, rhs), None
        return ir_builder.create_add_f(lhs, rhs), None

    def initial_build_ir(self, ir_builder, dt: DTDescriptor, first_ele: int) -> Tuple[int, int | None]:
        if isinstance(dt, IntegerDTDescriptor):
            return ir_builder.create_constant(0), None
        return ir_builder.create_float_cast(ir_builder.create_constant(0)), None
