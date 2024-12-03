from typing import Any

from pyzk.opdef.ndarray.abstract_aggregator import AbstractAggregator


class NDArray_AnyOp(AbstractAggregator):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "NDArray::any"

    @classmethod
    def get_name(cls) -> str:
        return "NDArray::any"

    def aggregator_func(self, lhs: Any, rhs: Any) -> Any:
        if lhs is not None and rhs is not None:
            return 1 if lhs != 0 or rhs != 0 else 0
        elif lhs is None and rhs is None:
            return None
        elif lhs is None and rhs is not None:
            return None if rhs == 0 else 1
        elif lhs is not None and rhs is None:
            return None if lhs == 0 else 1
        raise NotImplementedError()

    def initial_func(self) -> Any:
        return 0
