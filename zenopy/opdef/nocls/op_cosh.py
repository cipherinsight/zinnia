from typing import List, Dict, Optional

from zenopy.debug.exception import TypeInferenceError
from zenopy.opdef.nocls.abstract_op import AbstractOp
from zenopy.internal.dt_descriptor import IntegerDTDescriptor, FloatDTDescriptor, FloatType
from zenopy.debug.dbg_info import DebugInfo
from zenopy.builder.abstract_ir_builder import AbsIRBuilderInterface
from zenopy.builder.value import Value, IntegerValue, FloatValue, NDArrayValue


class CosHOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "cosh"

    @classmethod
    def get_name(cls) -> str:
        return "cosh"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("x")
        ]

    def build(self, reducer: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        x = kwargs["x"]
        if isinstance(x, IntegerValue):
            return reducer.ir_cosh_f(reducer.ir_float_cast(x))
        elif isinstance(x, FloatValue):
            return reducer.ir_cosh_f(x)
        elif isinstance(x, NDArrayValue) and isinstance(x.dtype(), IntegerDTDescriptor):
            return x.unary(FloatType, lambda u: reducer.ir_cosh_f(reducer.ir_float_cast(u)))
        elif isinstance(x, NDArrayValue) and isinstance(x.dtype(), FloatDTDescriptor):
            return x.unary(FloatType, lambda u: reducer.ir_cosh_f(u))
        raise TypeInferenceError(dbg, f"Operator `{self.get_name()}` not defined for `{x.type()}`")
