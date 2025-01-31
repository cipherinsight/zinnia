from typing import Dict, Optional, List

from zinnia.compile.builder.abstract_ir_builder import AbsIRBuilderInterface
from zinnia.compile.builder.value import Value, IntegerValue, FloatValue, NDArrayValue
from zinnia.compile.type_sys import FloatType, FloatDTDescriptor, IntegerDTDescriptor
from zinnia.debug.dbg_info import DebugInfo
from zinnia.debug.exception import TypeInferenceError
from zinnia.opdef.abstract.abstract_op import AbstractOp


class NP_CosOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "np.cos"

    @classmethod
    def get_name(cls) -> str:
        return "cos"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("x")
        ]

    def build(self, builder: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        x = kwargs["x"]
        if isinstance(x, IntegerValue):
            return builder.ir_cos_f(builder.ir_float_cast(x))
        elif isinstance(x, FloatValue):
            return builder.ir_cos_f(x)
        elif isinstance(x, NDArrayValue) and isinstance(x.dtype(), IntegerDTDescriptor):
            return x.unary(FloatType, lambda u: builder.ir_cos_f(builder.ir_float_cast(u)))
        elif isinstance(x, NDArrayValue) and isinstance(x.dtype(), FloatDTDescriptor):
            return x.unary(FloatType, lambda u: builder.ir_cos_f(u))
        raise TypeInferenceError(dbg, f"Operator `{self.get_name()}` not defined for `{x.type()}`")
