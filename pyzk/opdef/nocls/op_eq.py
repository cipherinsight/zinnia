from typing import Callable, Any

from pyzk.opdef.abstract_compare import AbstractCompare


class EqualOp(AbstractCompare):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "eq"

    @classmethod
    def get_name(cls) -> str:
        return "eq"

    def get_inference_op_lambda(self) -> Callable[[Any, Any], Any]:
        return lambda x, y: (1 if x == y else 0) if x is not None and y is not None else None
