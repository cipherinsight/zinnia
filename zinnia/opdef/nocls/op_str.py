import copy
from typing import List, Dict, Optional

from zinnia.debug.exception import TypeInferenceError
from zinnia.opdef.nocls.abstract_op import AbstractOp
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.builder.abstract_ir_builder import AbsIRBuilderInterface
from zinnia.compile.builder.value import Value, IntegerValue, FloatValue, NDArrayValue, TupleValue, ListValue, \
    StringValue


class StrOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "str"

    @classmethod
    def get_name(cls) -> str:
        return "str"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("x")
        ]

    def build(self, builder: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        x = kwargs["x"]
        if isinstance(x, IntegerValue):
            return builder.ir_str_i(x)
        elif isinstance(x, FloatValue):
            return builder.ir_str_f(x)
        elif isinstance(x, NDArrayValue):
            return builder.op_str(builder.op_ndarray_tolist(x))
        elif isinstance(x, TupleValue):
            str_values = [builder.op_str(v) for v in x.values()]
            begin = builder.ir_constant_str("(")
            end = builder.ir_constant_str(")")
            inner = builder.ir_constant_str("")
            sep = builder.ir_constant_str(", ")
            for i, value in enumerate(str_values):
                inner = builder.op_add(inner, value)
                if i < len(str_values) - 1:
                    inner = builder.op_add(inner, sep)
            return builder.op_add(builder.op_add(begin, inner), end)
        elif isinstance(x, ListValue):
            str_values = [builder.op_str(v) for v in x.values()]
            begin = builder.ir_constant_str("[")
            end = builder.ir_constant_str("]")
            inner = builder.ir_constant_str("")
            sep = builder.ir_constant_str(", ")
            for i, value in enumerate(str_values):
                inner = builder.op_add(inner, value)
                if i < len(str_values) - 1:
                    inner = builder.op_add(inner, sep)
            return builder.op_add(builder.op_add(begin, inner), end)
        elif isinstance(x, StringValue):
            return copy.copy(x)
        raise TypeInferenceError(dbg, f"Operator `{self.get_name()}` not defined for `{x.type()}`")
