import copy
from typing import List, Dict, Optional

from zinnia.debug.exception import TypeInferenceError
from zinnia.opdef.abstract.abstract_op import AbstractOp
from zinnia.compile.type_sys import FloatType, IntegerType
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.builder.abstract_ir_builder import AbsIRBuilderInterface
from zinnia.compile.builder.value import IntegerValue, FloatValue, Value, NDArrayValue, ListValue, TupleValue


class IntCastOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "int"

    @classmethod
    def get_name(cls) -> str:
        return "int"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("x")
        ]

    def build(self, builder: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        x = kwargs["x"]
        if isinstance(x, IntegerValue):
            return copy.copy(x)
        elif isinstance(x, FloatValue):
            return builder.ir_int_cast(x)
        elif isinstance(x, NDArrayValue):
            flattened = x.flattened_values()
            if len(flattened) != 1:
                raise TypeInferenceError(dbg, f'Only length-1 arrays can be converted to scalars')
            if x.dtype() == IntegerType:
                return copy.copy(flattened[0])
            elif x.dtype() == FloatType:
                return builder.ir_int_cast(flattened[0])
            raise NotImplementedError()
        elif isinstance(x, ListValue):
            raise TypeInferenceError(dbg, f'List cannot be converted to integer scalars')
        elif isinstance(x, TupleValue):
            raise TypeInferenceError(dbg, f'Tuple cannot be converted to integer scalars')
        raise TypeInferenceError(dbg, f"Unsupported argument type for `{self.get_name()}`: {x.type()}")
