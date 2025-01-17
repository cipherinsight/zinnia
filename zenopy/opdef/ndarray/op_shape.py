from typing import List, Dict, Optional

from zenopy.opdef.nocls.abstract_op import AbstractOp
from zenopy.internal.dt_descriptor import IntegerType
from zenopy.debug.dbg_info import DebugInfo
from zenopy.builder.abstract_ir_builder import AbsIRBuilderInterface
from zenopy.builder.value import Value, NDArrayValue, TupleValue


class NDArray_ShapeOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "NDArray::shape"

    @classmethod
    def get_name(cls) -> str:
        return "NDArray::shape"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("self")
        ]

    def build(self, reducer: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        the_self = kwargs["self"]
        assert isinstance(the_self, NDArrayValue)
        shape = the_self.shape()
        return TupleValue(tuple(IntegerType for _ in shape), tuple(reducer.ir_constant_int(x) for x in shape))
