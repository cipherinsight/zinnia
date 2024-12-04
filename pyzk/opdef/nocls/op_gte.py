from typing import Callable, Any

from pyzk.opdef.abstract_compare import AbstractCompare


class GreaterThanOrEqualOp(AbstractCompare):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "gte"

    @classmethod
    def get_name(cls) -> str:
        return "gte"

    def get_inference_op_lambda(self) -> Callable[[Any, Any], Any]:
        return lambda x, y: (1 if x >= y else 0) if x is not None and y is not None else None

    def get_flatten_op_lambda(self, ir_builder) -> Callable[[int, int], int]:
        return lambda x, y: ir_builder.create_greater_than_or_equal(x, y)
