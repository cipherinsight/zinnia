import copy
from typing import List, Dict, Optional

from zenopy.debug.exception import TypeInferenceError
from zenopy.internal.dt_descriptor import IntegerType, FloatType
from zenopy.opdef.nocls.abstract_op import AbstractOp
from zenopy.debug.dbg_info import DebugInfo
from zenopy.builder.abstract_ir_builder import AbsIRBuilderInterface
from zenopy.builder.value import Value, IntegerValue, FloatValue, NDArrayValue, ListValue, TupleValue


class FloatScalarOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "float_scalar"

    @classmethod
    def get_name(cls) -> str:
        return "float_scalar"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("x")
        ]

    def build(self, reducer: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        x = kwargs["x"]
        if isinstance(x, IntegerValue):
            return reducer.ir_float_cast(x)
        elif isinstance(x, FloatValue):
            return copy.copy(x)
        elif isinstance(x, NDArrayValue):
            flattened = x.flattened_values()
            if len(flattened) != 1:
                raise TypeInferenceError(dbg, f'Only length-1 arrays can be converted to scalars')
            if x.dtype() == FloatType:
                return copy.copy(flattened[0])
            elif x.dtype() == IntegerType:
                return reducer.ir_float_cast(flattened[0])
            raise NotImplementedError()
        elif isinstance(x, ListValue):
            raise TypeInferenceError(dbg, f'List cannot be converted to float scalars')
        elif isinstance(x, TupleValue):
            raise TypeInferenceError(dbg, f'Tuple cannot be converted to float scalars')
        raise TypeInferenceError(dbg, f"Unsupported argument type for `{self.get_name()}`: {x.type()}")
