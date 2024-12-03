from typing import Callable, Any

from pyzk.opdef.abstract_arithemetic import AbstractArithemetic


class DivOp(AbstractArithemetic):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "div"

    @classmethod
    def get_name(cls) -> str:
        return "div"

    def get_inference_op_lambda(self) -> Callable[[Any, Any], Any]:
        return lambda x, y: None
