import copy
from typing import List, Dict, Optional

from zinnia.debug.exception import TypeInferenceError
from zinnia.compile.type_sys import FloatType, IntegerType
from zinnia.opdef.abstract.abstract_op import AbstractOp
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.builder.abstract_ir_builder import AbsIRBuilderInterface
from zinnia.compile.builder.value import Value, IntegerValue, FloatValue, NDArrayValue, ListValue, TupleValue


class FloatCastOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "float"

    @classmethod
    def get_name(cls) -> str:
        return "float"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("x")
        ]

    def build(self, builder: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        x = kwargs["x"]
        if isinstance(x, IntegerValue):
            return builder.ir_float_cast(x)
        elif isinstance(x, FloatValue):
            return copy.copy(x)
        elif isinstance(x, NDArrayValue):
            flattened = x.flattened_values()
            if len(flattened) != 1:
                raise TypeInferenceError(dbg, f'Only length-1 arrays can be converted to scalars')
            if x.dtype() == FloatType:
                return copy.copy(flattened[0])
            elif x.dtype() == IntegerType:
                return builder.ir_float_cast(flattened[0])
            raise NotImplementedError()
        elif isinstance(x, ListValue):
            raise TypeInferenceError(dbg, f'List cannot be converted to float scalars')
        elif isinstance(x, TupleValue):
            raise TypeInferenceError(dbg, f'Tuple cannot be converted to float scalars')
        raise TypeInferenceError(dbg, f"Unsupported argument type for `{self.get_name()}`: {x.type()}")
