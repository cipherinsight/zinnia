from typing import Any

from pyzk.opdef.ndarray.abstract_aggregator import AbstractAggregator


class NDArray_SumOp(AbstractAggregator):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "NDArray::sum"

    @classmethod
    def get_name(cls) -> str:
        return "NDArray::sum"

    def aggregator_func(self, lhs: Any, rhs: Any) -> Any:
        return (lhs + rhs) if lhs is not None and rhs is not None else None

    def initial_func(self) -> Any:
        return 0

    def aggregator_build_ir(self, ir_builder, lhs: int, rhs: int) -> int:
        return ir_builder.create_add(lhs, rhs)

    def initial_build_ir(self, ir_builder) -> int:
        return ir_builder.create_constant(0)
