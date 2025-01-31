from typing import List, Dict, Optional

from zinnia.debug.exception import TypeInferenceError
from zinnia.opdef.abstract.abstract_op import AbstractOp
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.builder.abstract_ir_builder import AbsIRBuilderInterface
from zinnia.compile.builder.value import Value, IntegerValue, FloatValue, NDArrayValue


class Math_SinHOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "math.sinh"

    @classmethod
    def get_name(cls) -> str:
        return "sinh"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("x")
        ]

    def build(self, builder: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        x = kwargs["x"]
        if isinstance(x, IntegerValue) or isinstance(x, FloatValue) or isinstance(x, NDArrayValue):
            return builder.ir_sinh_f(builder.op_float_cast(x))
        raise TypeInferenceError(dbg, f"Operator `{self.get_name()}` not defined for `{x.type()}`")
