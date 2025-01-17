from typing import Callable

from zenopy.opdef.nocls.abstract_integer_arithemetic import AbstractIntegerArithemetic
from zenopy.builder.abstract_ir_builder import AbsIRBuilderInterface
from zenopy.builder.value import IntegerValue


class LogicalAndOp(AbstractIntegerArithemetic):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "logical_and"

    @classmethod
    def get_name(cls) -> str:
        return "logical_and"

    def get_reduce_op_lambda(self, reducer: AbsIRBuilderInterface) -> Callable[[IntegerValue, IntegerValue], IntegerValue]:
        return lambda x, y: reducer.ir_logical_and(x, y)
