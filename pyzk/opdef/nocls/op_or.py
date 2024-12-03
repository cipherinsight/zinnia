from typing import Any

from pyzk.opdef.abstract_binary_logical import AbstractBinaryLogical


class OrOp(AbstractBinaryLogical):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "or"

    @classmethod
    def get_name(cls) -> str:
        return "or"

    def perform_inference(self, lhs: Any, rhs: Any) -> Any:
        if lhs is not None and rhs is not None:
            return 1 if lhs != 0 or rhs != 0 else 0
        elif lhs is None and rhs is None:
            return None
        elif lhs is None and rhs is not None:
            return None if rhs == 0 else 1
        elif lhs is not None and rhs is None:
            return None if lhs == 0 else 1
        raise NotImplementedError()
