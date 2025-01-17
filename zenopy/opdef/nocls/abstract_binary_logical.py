from typing import List, Dict, Optional

from zenopy.debug.exception import TypeInferenceError
from zenopy.opdef.nocls.abstract_op import AbstractOp
from zenopy.debug.dbg_info import DebugInfo
from zenopy.builder.abstract_ir_builder import AbsIRBuilderInterface
from zenopy.builder.value import Value, IntegerValue


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

    def perform_reduce(self, lhs: Value, rhs: Value) -> Value:
        raise NotImplementedError()

    def build(self, reducer: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        lhs, rhs = kwargs["lhs"], kwargs["rhs"]
        if isinstance(lhs, IntegerValue) and isinstance(rhs, IntegerValue):
            return self.perform_reduce(lhs, rhs)
        raise TypeInferenceError(dbg, f'Invalid binary logical operator `{self.get_signature()}` on operands {lhs} and {rhs}, as they must be integer values')
