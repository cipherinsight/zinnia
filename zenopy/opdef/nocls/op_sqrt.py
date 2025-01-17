import math
from typing import List, Dict, Optional, Any

from zenopy.debug.exception import TypeInferenceError, StaticInferenceError
from zenopy.internal.dt_descriptor import IntegerType, FloatType
from zenopy.opdef.nocls.abstract_op import AbstractOp
from zenopy.debug.dbg_info import DebugInfo
from zenopy.builder.abstract_ir_builder import AbsIRBuilderInterface
from zenopy.builder.value import Value, IntegerValue, FloatValue, NDArrayValue


class SqrtOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "sqrt"

    @classmethod
    def get_name(cls) -> str:
        return "sqrt"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("x")
        ]

    def build(self, reducer: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        x = kwargs["x"]
        if isinstance(x, IntegerValue):
            return reducer.ir_sqrt(reducer.ir_float_cast(x), dbg=dbg)
        elif isinstance(x, FloatValue):
            return reducer.ir_sqrt(x, dbg=dbg)
        elif isinstance(x, NDArrayValue) and x.dtype() == IntegerType:
            return x.unary(IntegerType, lambda v: reducer.ir_sqrt(reducer.ir_float_cast(v), dbg=dbg))
        elif isinstance(x, NDArrayValue) and x.dtype() == FloatType:
            return x.unary(IntegerType, lambda v: reducer.ir_sqrt(v, dbg=dbg))
        raise TypeInferenceError(dbg, f"Unsupported argument type for `{self.get_name()}`: {x.type()}")
