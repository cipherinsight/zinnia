from typing import Callable

from zinnia.op_def.abstract.abstract_integer_arithemetic import AbstractLogicalArithemetic
from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import IntegerValue


class LogicalAndOp(AbstractLogicalArithemetic):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "logical_and"

    @classmethod
    def get_name(cls) -> str:
        return "logical_and"

    def get_build_op_lambda(self, builder: IRBuilderInterface) -> Callable[[IntegerValue, IntegerValue], IntegerValue]:
        def _inner(x, y):
            if isinstance(x, IntegerValue):
                x = builder.op_bool_cast(x)
            if isinstance(y, IntegerValue):
                y = builder.op_bool_cast(y)
            return builder.ir_logical_and(x, y)
        return _inner
