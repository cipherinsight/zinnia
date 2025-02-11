import copy
from typing import List, Dict, Optional

from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.debug.dbg_info import DebugInfo
from zinnia.debug.exception import TypeInferenceError
from zinnia.compile.type_sys import IntegerType, FloatType
from zinnia.op_def.abstract.abstract_op import AbstractOp
from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import Value, IntegerValue, FloatValue, NDArrayValue


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

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
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
