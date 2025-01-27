from typing import Callable

from zinnia.opdef.nocls.abstract_integer_arithemetic import AbstractIntegerArithemetic
from zinnia.compile.builder.abstract_ir_builder import AbsIRBuilderInterface
from zinnia.compile.builder.value import IntegerValue


class LogicalAndOp(AbstractIntegerArithemetic):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "logical_and"

    @classmethod
    def get_name(cls) -> str:
        return "logical_and"

    def get_build_op_lambda(self, builder: AbsIRBuilderInterface) -> Callable[[IntegerValue, IntegerValue], IntegerValue]:
        return lambda x, y: builder.ir_logical_and(x, y)
