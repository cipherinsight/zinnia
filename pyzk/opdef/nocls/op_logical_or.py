from typing import Callable

from pyzk.opdef.nocls.abstract_integer_arithemetic import AbstractIntegerArithemetic
from pyzk.builder.abstract_ir_builder import AbsIRBuilderInterface
from pyzk.builder.value import IntegerValue


class LogicalOrOp(AbstractIntegerArithemetic):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "logical_or"

    @classmethod
    def get_name(cls) -> str:
        return "logical_or"

    def get_reduce_op_lambda(self, reducer: AbsIRBuilderInterface) -> Callable[[IntegerValue, IntegerValue], IntegerValue]:
        return lambda x, y: reducer.ir_logical_or(x, y)
