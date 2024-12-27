from typing import Any, Tuple

from pyzk.internal.dt_descriptor import DTDescriptor, IntegerDTDescriptor
from pyzk.opdef.ndarray.abstract_aggregator import AbstractAggregator


class NDArray_AnyOp(AbstractAggregator):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "NDArray::any"

    @classmethod
    def get_name(cls) -> str:
        return "NDArray::any"

    def get_result_dtype(self, element_dt: DTDescriptor):
        return IntegerDTDescriptor()

    def is_allowed_ndarray_dtype(self, element_dt: DTDescriptor) -> bool:
        return isinstance(element_dt, IntegerDTDescriptor)

    def aggregator_func(self, lhs: Any, lhs_i: int, rhs: Any, rhs_i: int, dt: DTDescriptor) -> Tuple[Any, int | None]:
        if lhs is not None and rhs is not None:
            return 1 if lhs != 0 or rhs != 0 else 0, None
        elif lhs is None and rhs is None:
            return None, None
        elif lhs is None and rhs is not None:
            return None if rhs == 0 else 1, None
        elif lhs is not None and rhs is None:
            return None if lhs == 0 else 1, None
        raise NotImplementedError()

    def initial_func(self, dt: DTDescriptor, first_ele: Any) -> Tuple[Any, int | None]:
        return 0, None

    def aggregator_build_ir(self, ir_builder, lhs: int, lhs_i: int, rhs: int, rhs_i: int, dt: DTDescriptor) -> Tuple[int, int | None]:
        return ir_builder.create_logical_or(lhs, rhs), None

    def initial_build_ir(self, ir_builder, dt: DTDescriptor, first_ele: int) -> Tuple[int, int | None]:
        return ir_builder.create_constant(0), None
