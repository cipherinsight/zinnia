from typing import List, Dict, Optional

from pyzk.debug.exception import TypeInferenceError
from pyzk.internal.dt_descriptor import NDArrayDTDescriptor
from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.debug.dbg_info import DebugInfo
from pyzk.builder.abstract_ir_builder import AbsIRBuilderInterface
from pyzk.builder.value import Value, ListValue, TupleValue, NDArrayValue


class ListOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "list"

    @classmethod
    def get_name(cls) -> str:
        return "list"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("x")
        ]

    def build(self, reducer: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        x = kwargs["x"]
        if isinstance(x, ListValue):
            return ListValue(list(x.types()), list(x.values()))
        elif isinstance(x, TupleValue):
            return ListValue(list(x.types()), list(x.values()))
        elif isinstance(x, NDArrayValue):
            sub_element_shape = x.shape()[1:]
            sub_element_type = NDArrayDTDescriptor(sub_element_shape, x.dtype())
            return ListValue(
                list(sub_element_type for _ in range(x.shape()[0])),
                list(reducer.op_get_item(x, reducer.op_square_brackets([reducer.ir_constant_int(i)])) for i in range(x.shape()[0]))
            )
        raise TypeInferenceError(dbg, f"`list` operator is not defined on {x.type()}")
