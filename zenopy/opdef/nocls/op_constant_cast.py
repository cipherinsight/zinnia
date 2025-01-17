from typing import List, Dict, Optional

from zenopy.debug.exception import StaticInferenceError, TypeInferenceError
from zenopy.opdef.nocls.abstract_op import AbstractOp
from zenopy.debug.dbg_info import DebugInfo
from zenopy.builder.abstract_ir_builder import AbsIRBuilderInterface
from zenopy.builder.value import Value, IntegerValue, FloatValue


class ConstantCastOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "constant_cast"

    @classmethod
    def get_name(cls) -> str:
        return "constant_cast"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("x")
        ]

    def build(self, reducer: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        x = kwargs["x"]
        if isinstance(x, IntegerValue):
            if x.val() is None:
                raise StaticInferenceError(dbg, 'Cannot statically infer the corresponding value')
            return x
        elif isinstance(x, FloatValue):
            if x.val() is None:
                raise StaticInferenceError(dbg, 'Cannot statically infer the corresponding value')
            return x
        raise TypeInferenceError(dbg, f"Operator `{self.get_name()}` not defined on `{x.type()}`")
