from typing import Any

from pyzk.internal.dt_descriptor import DTDescriptor, IntegerDTDescriptor
from pyzk.opdef.ndarray.abstract_aggregator import AbstractAggregator


class NDArray_AllOp(AbstractAggregator):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "NDArray::all"

    @classmethod
    def get_name(cls) -> str:
        return "NDArray::all"

    def get_result_dtype(self, element_dt: DTDescriptor):
        return IntegerDTDescriptor()

    def is_allowed_ndarray_dtype(self, element_dt: DTDescriptor) -> bool:
        return isinstance(element_dt, IntegerDTDescriptor)

    def aggregator_func(self, lhs: Any, rhs: Any, dt: DTDescriptor) -> Any:
        if lhs is not None and rhs is not None:
            return 1 if lhs != 0 and rhs != 0 else 0
        elif lhs is None and rhs is None:
            return None
        elif lhs is None and rhs is not None:
            return None if rhs != 0 else 0
        elif lhs is not None and rhs is None:
            return None if lhs != 0 else 0
        raise NotImplementedError()

    def initial_func(self, dt: DTDescriptor) -> Any:
        return 1

    def aggregator_build_ir(self, ir_builder, lhs: int, rhs: int, dt: DTDescriptor) -> int:
        return ir_builder.create_logical_and(lhs, rhs)

    def initial_build_ir(self, ir_builder, dt: DTDescriptor) -> int:
        return ir_builder.create_constant(1)
