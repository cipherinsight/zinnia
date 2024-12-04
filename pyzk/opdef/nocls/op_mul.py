from typing import Callable, Any

from pyzk.opdef.abstract_arithemetic import AbstractArithemetic


class MulOp(AbstractArithemetic):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "mul"

    @classmethod
    def get_name(cls) -> str:
        return "mul"

    def get_inference_op_lambda(self) -> Callable[[Any, Any], Any]:
        return lambda x, y: x * y if x is not None and y is not None else None

    def get_flatten_op_lambda(self, ir_builder) -> Callable[[int, int], int]:
        return lambda x, y: ir_builder.create_mul(x, y)
