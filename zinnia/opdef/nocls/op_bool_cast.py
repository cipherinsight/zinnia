from typing import List, Dict, Optional

from zinnia.debug.exception import TypeInferenceError
from zinnia.opdef.abstract.abstract_op import AbstractOp
from zinnia.compile.type_sys import IntegerType, FloatType
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.builder.abstract_ir_builder import AbsIRBuilderInterface
from zinnia.compile.builder.value import Value, IntegerValue, NDArrayValue, FloatValue, ListValue, TupleValue


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
            flattened = x.flattened_values()
            if len(flattened) != 1:
                raise TypeInferenceError(dbg, f'The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()')
            if x.dtype() == IntegerType:
                return builder.ir_not_equal_i(flattened[0], builder.ir_constant_int(0))
            elif x.dtype() == FloatType:
                return builder.ir_not_equal_f(flattened[0], builder.ir_constant_float(0.0))
            raise NotImplementedError()
        elif isinstance(x, ListValue) or isinstance(x, TupleValue):
            if len(x.types()) > 0:
                return builder.ir_constant_int(1)
            return builder.ir_constant_int(0)
        raise TypeInferenceError(dbg, f"Unsupported argument type for `{self.get_name()}`: {x.type()}")
