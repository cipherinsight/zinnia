from typing import Callable, Any

from pyzk.opdef.nocls.abstract_arithemetic import AbstractArithemetic


class SubOp(AbstractArithemetic):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "sub"

    @classmethod
    def get_name(cls) -> str:
        return "sub"

    def get_inference_op_lambda(self) -> Callable[[Any, Any], Any]:
        return lambda x, y: x - y if x is not None and y is not None else None

    def get_flatten_op_lambda(self, ir_builder) -> Callable[[int, int], int]:
        return lambda x, y: ir_builder.create_sub(x, y)
