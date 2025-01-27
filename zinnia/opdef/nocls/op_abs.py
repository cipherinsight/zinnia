from typing import List, Dict, Optional

from zinnia.debug.exception import TypeInferenceError
from zinnia.opdef.nocls.abstract_op import AbstractOp
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.builder.abstract_ir_builder import AbsIRBuilderInterface
from zinnia.compile.builder.value import Value, IntegerValue, FloatValue, NDArrayValue


class AbsOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "abs"

    @classmethod
    def get_name(cls) -> str:
        return "abs"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("x")
        ]

    def build(self, builder: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        x = kwargs["x"]
        if isinstance(x, IntegerValue):
            return builder.ir_abs_i(x)
        elif isinstance(x, FloatValue):
            return builder.ir_abs_f(x)
        elif isinstance(x, NDArrayValue):
            return x.unary(x.dtype(), lambda u: builder.op_abs(u))
        raise TypeInferenceError(dbg, f'Operator `{self.get_signature()}` not defined on type `{x.type()}`')
