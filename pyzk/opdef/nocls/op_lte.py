from typing import Callable, Any

from pyzk.opdef.abstract_compare import AbstractCompare


class LessThanOrEqualOp(AbstractCompare):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "lte"

    @classmethod
    def get_name(cls) -> str:
        return "lte"

    def get_inference_op_lambda(self) -> Callable[[Any, Any], Any]:
        return lambda x, y: (1 if x <= y else 0) if x is not None and y is not None else None
