from typing import List, Dict, Optional

from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.debug.exception import TypeInferenceError
from zinnia.op_def.abstract.abstract_op import AbstractOp
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import Value, IntegerValue, FloatValue, NDArrayValue


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

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        x = kwargs["x"]
        if isinstance(x, IntegerValue):
            return builder.ir_abs_i(x)
        elif isinstance(x, FloatValue):
            return builder.ir_abs_f(x)
        elif isinstance(x, NDArrayValue):
            return x.unary(x.dtype(), lambda u: builder.op_abs(u))
        raise TypeInferenceError(dbg, f'Operator `{self.get_signature()}` not defined on type `{x.type()}`')
