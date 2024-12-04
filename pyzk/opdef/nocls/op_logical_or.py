from typing import Callable, Any

from pyzk.opdef.nocls.abstract_arithemetic import AbstractArithemetic


class LogicalOrOp(AbstractArithemetic):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "logical_or"

    @classmethod
    def get_name(cls) -> str:
        return "logical_or"

    def get_inference_op_lambda(self) -> Callable[[Any, Any], Any]:
        def _inner(lhs: Any, rhs: Any) -> Any:
            if lhs is not None and rhs is not None:
                return 1 if lhs != 0 or rhs != 0 else 0
            elif lhs is None and rhs is None:
                return None
            elif lhs is None and rhs is not None:
                return None if rhs == 0 else 1
            elif lhs is not None and rhs is None:
                return None if lhs == 0 else 1
            raise NotImplementedError()
        return _inner

    def get_flatten_op_lambda(self, ir_builder) -> Callable[[int, int], int]:
        return lambda x, y: ir_builder.create_logical_or(x, y)
