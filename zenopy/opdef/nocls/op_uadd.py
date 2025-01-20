import copy
from typing import List, Dict, Optional

from zenopy.debug.dbg_info import DebugInfo
from zenopy.debug.exception import TypeInferenceError
from zenopy.internal.dt_descriptor import IntegerType, FloatType
from zenopy.opdef.nocls.abstract_op import AbstractOp
from zenopy.builder.abstract_ir_builder import AbsIRBuilderInterface
from zenopy.builder.value import Value, IntegerValue, FloatValue, NDArrayValue


class UAddOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "uadd"

    @classmethod
    def get_name(cls) -> str:
        return "uadd"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("x")
        ]

    def build(self, reducer: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        x = kwargs["x"]
        if isinstance(x, IntegerValue):
            return copy.copy(x)
        elif isinstance(x, FloatValue):
            return copy.copy(x)
        elif isinstance(x, NDArrayValue) and x.dtype() == IntegerType:
            return copy.deepcopy(x)
        elif isinstance(x, NDArrayValue) and x.dtype() == FloatType:
            return copy.deepcopy(x)
        raise TypeInferenceError(dbg, f"Unsupported argument type for `{self.get_name()}`: {x.type()}")
