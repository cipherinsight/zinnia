from typing import List, Dict, Optional

from zinnia.debug.dbg_info import DebugInfo
from zinnia.debug.exception import TypeInferenceError
from zinnia.compile.type_sys import IntegerType, FloatType
from zinnia.opdef.abstract.abstract_op import AbstractOp
from zinnia.compile.builder.abstract_ir_builder import AbsIRBuilderInterface
from zinnia.compile.builder.value import Value, IntegerValue, FloatValue, NDArrayValue, TupleValue, ListValue, NoneValue


class ExposePublicOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "expose_public"

    @classmethod
    def get_name(cls) -> str:
        return "expose_public"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("x")
        ]

    def build(self, builder: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        x = kwargs["x"]
        if isinstance(x, IntegerValue):
            return builder.ir_expose_public_i(x)
        elif isinstance(x, FloatValue):
            return builder.ir_expose_public_f(x)
        elif isinstance(x, TupleValue) or isinstance(x, ListValue):
            for val in x.values():
                builder.op_expose_public(val)
            return NoneValue()
        elif isinstance(x, NDArrayValue) and x.dtype() == IntegerType:
            for v in x.flattened_values():
                builder.ir_expose_public_i(v)
            return NoneValue()
        elif isinstance(x, NDArrayValue) and x.dtype() == FloatType:
            for v in x.flattened_values():
                builder.ir_expose_public_f(v)
            return NoneValue()
        raise TypeInferenceError(dbg, f"Unsupported argument type for `{self.get_name()}`: {x.type()}")
