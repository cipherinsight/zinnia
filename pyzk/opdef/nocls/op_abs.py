from typing import List, Dict, Optional

from pyzk.debug.exception import TypeInferenceError
from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.debug.dbg_info import DebugInfo
from pyzk.builder.abstract_ir_builder import AbsIRBuilderInterface
from pyzk.builder.value import Value, IntegerValue, FloatValue, NDArrayValue


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

    def build(self, reducer: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        x = kwargs["x"]
        if isinstance(x, IntegerValue):
            return reducer.ir_abs_i(x)
        elif isinstance(x, FloatValue):
            return reducer.ir_abs_f(x)
        elif isinstance(x, NDArrayValue):
            return x.unary(x.dtype(), lambda u: reducer.op_abs(u))
        raise TypeInferenceError(dbg, f'Operator `{self.get_signature()}` not defined on type `{x.type()}`')
