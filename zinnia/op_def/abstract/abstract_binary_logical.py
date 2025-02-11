from typing import List, Dict, Optional

from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.debug.exception import TypeInferenceError
from zinnia.op_def.abstract.abstract_op import AbstractOp
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import Value, IntegerValue


class AbstractBinaryLogical(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        raise NotImplementedError()

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("lhs"),
            AbstractOp._ParamEntry("rhs"),
        ]

    def perform_build(self, lhs: Value, rhs: Value) -> Value:
        raise NotImplementedError()

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        lhs, rhs = kwargs["lhs"], kwargs["rhs"]
        if isinstance(lhs, IntegerValue) and isinstance(rhs, IntegerValue):
            return self.perform_build(lhs, rhs)
        raise TypeInferenceError(dbg, f'Invalid binary logical operator `{self.get_signature()}` on operands {lhs} and {rhs}, as they must be integer values')
