from typing import List, Dict, Optional

from zenopy.debug.exception import TypeInferenceError
from zenopy.opdef.nocls.abstract_op import AbstractOp
from zenopy.internal.dt_descriptor import IntegerType, FloatType
from zenopy.debug.dbg_info import DebugInfo
from zenopy.builder.abstract_ir_builder import AbsIRBuilderInterface
from zenopy.builder.value import IntegerValue, NDArrayValue, FloatValue, Value


class ExpOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "exp"

    @classmethod
    def get_name(cls) -> str:
        return "exp"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("x")
        ]

    def build(self, reducer: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        x = kwargs["x"]
        if isinstance(x, IntegerValue):
            return reducer.ir_exp_f(reducer.ir_float_cast(x))
        elif isinstance(x, FloatValue):
            return reducer.ir_exp_f(x)
        elif isinstance(x, NDArrayValue) and x.dtype() == IntegerType:
            return x.unary(IntegerType, lambda v: reducer.ir_exp_f(reducer.ir_float_cast(v)))
        elif isinstance(x, NDArrayValue) and x.dtype() == FloatType:
            return x.unary(IntegerType, lambda v: reducer.ir_exp_f(v))
        raise TypeInferenceError(dbg, f"Unsupported argument type for `{self.get_name()}`: {x.type()}")
