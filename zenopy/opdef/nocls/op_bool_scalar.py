from typing import List, Dict, Optional

from zenopy.debug.exception import TypeInferenceError
from zenopy.internal.dt_descriptor import IntegerType, FloatType
from zenopy.opdef.nocls.abstract_op import AbstractOp
from zenopy.debug.dbg_info import DebugInfo
from zenopy.builder.abstract_ir_builder import AbsIRBuilderInterface
from zenopy.builder.value import Value, IntegerValue, FloatValue, NDArrayValue, ListValue, TupleValue


class BoolScalarOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "bool_scalar"

    @classmethod
    def get_name(cls) -> str:
        return "bool_scalar"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("x")
        ]

    def build(self, reducer: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        x = kwargs["x"]
        if isinstance(x, IntegerValue):
            return reducer.ir_not_equal_i(x, reducer.ir_constant_int(0))
        elif isinstance(x, FloatValue):
            return reducer.ir_not_equal_f(x, reducer.ir_constant_float(0.0))
        elif isinstance(x, NDArrayValue):
            flattened = x.flattened_values()
            if len(flattened) != 1:
                raise TypeInferenceError(dbg, f'The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()')
            if x.dtype() == IntegerType:
                return reducer.ir_not_equal_i(flattened[0], reducer.ir_constant_int(0))
            elif x.dtype() == FloatType:
                return reducer.ir_not_equal_f(flattened[0], reducer.ir_constant_float(0.0))
            raise NotImplementedError()
        elif isinstance(x, ListValue) or isinstance(x, TupleValue):
            if len(x.types()) > 0:
                return reducer.ir_constant_int(1)
            return reducer.ir_constant_int(0)
        raise TypeInferenceError(dbg, f"Unsupported argument type for `{self.get_name()}`: {x.type()}")
