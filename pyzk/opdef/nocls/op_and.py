from typing import Any

from pyzk.opdef.nocls.abstract_binary_logical import AbstractBinaryLogical


class AndOp(AbstractBinaryLogical):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "and"

    @classmethod
    def get_name(cls) -> str:
        return "and"

    def perform_inference(self, lhs: Any, rhs: Any) -> Any:
        if lhs is not None and rhs is not None:
            return 1 if lhs != 0 and rhs != 0 else 0
        elif lhs is None and rhs is None:
            return None
        elif lhs is None and rhs is not None:
            return None if rhs != 0 else 0
        elif lhs is not None and rhs is None:
            return None if lhs != 0 else 0
        raise NotImplementedError()

    def perform_flatten(self, ir_builder, lhs: Any, rhs: Any) -> Any:
        return ir_builder.create_logical_and(lhs, rhs)
