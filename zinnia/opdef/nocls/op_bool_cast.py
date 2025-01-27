from typing import List, Dict, Optional

from zinnia.debug.exception import TypeInferenceError
from zinnia.opdef.nocls.abstract_op import AbstractOp
from zinnia.compile.type_sys import IntegerType, FloatType
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.builder.abstract_ir_builder import AbsIRBuilderInterface
from zinnia.compile.builder.value import Value, IntegerValue, NDArrayValue, FloatValue


class BoolCastOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "bool_cast"

    @classmethod
    def get_name(cls) -> str:
        return "bool_cast"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("x")
        ]

    def build(self, builder: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        x = kwargs["x"]
        if isinstance(x, IntegerValue):
            return builder.ir_not_equal_i(x, builder.ir_constant_int(0))
        elif isinstance(x, FloatValue):
            return builder.ir_not_equal_f(x, builder.ir_constant_float(0.0))
        elif isinstance(x, NDArrayValue):
            if x.dtype() == IntegerType:
                return x.unary(IntegerType, lambda u: builder.ir_bool_cast(u))
            elif x.dtype() == FloatType:
                return x.unary(FloatType, lambda u: builder.ir_bool_cast(u))
        raise TypeInferenceError(dbg, f'Invalid `{self.get_signature()}` on operand {x.type()}')
