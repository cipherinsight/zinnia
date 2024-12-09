from typing import Any

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

    def aggregator_func(self, lhs: Any, rhs: Any, dt: DTDescriptor) -> Any:
        if isinstance(dt, IntegerDTDescriptor):
            return (lhs + rhs) if lhs is not None and rhs is not None else None
        return None

    def initial_func(self, dt: DTDescriptor) -> Any:
        if isinstance(dt, IntegerDTDescriptor):
            return 0
        return 0.0

    def aggregator_build_ir(self, ir_builder, lhs: int, rhs: int, dt: DTDescriptor) -> int:
        if isinstance(dt, IntegerDTDescriptor):
            return ir_builder.create_add_i(lhs, rhs)
        return ir_builder.create_add_f(lhs, rhs)

    def initial_build_ir(self, ir_builder, dt: DTDescriptor) -> int:
        if isinstance(dt, IntegerDTDescriptor):
            return ir_builder.create_constant(0)
        return ir_builder.create_float_cast(ir_builder.create_constant(0))
