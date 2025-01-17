from typing import List, Dict, Optional, Any

from pyzk.debug.exception import TypeInferenceError
from pyzk.internal.dt_descriptor import NDArrayDTDescriptor
from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.debug.dbg_info import DebugInfo
from pyzk.builder.abstract_ir_builder import AbsIRBuilderInterface
from pyzk.builder.value import Value, NDArrayValue, TupleValue, ListValue


class TupleOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "tuple"

    @classmethod
    def get_name(cls) -> str:
        return "tuple"

    def argparse(self, dbg_i: Optional[DebugInfo], args: List[Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        if len(kwargs) != 0:
            raise TypeInferenceError(dbg_i, "`tuple` operator does not support keyword arguments")
        if len(args) > 1:
            raise TypeInferenceError(dbg_i, f"`tuple` expected at most 1 argument, got {len(args)}")
        if len(args) == 0:
            return {}
        return {"x": args[0]}

    def build(self, reducer: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        if "x" not in kwargs:
            return TupleValue(tuple(), tuple())
        x = kwargs["x"]
        if isinstance(x, NDArrayValue):
            sub_element_shape = x.shape()[1:]
            sub_element_type = NDArrayDTDescriptor(sub_element_shape, x.dtype())
            return TupleValue(
                tuple(sub_element_type for _ in range(x.shape()[0])),
                tuple(reducer.op_get_item(x, reducer.op_square_brackets([reducer.ir_constant_int(i)])) for i in range(x.shape()[0]))
            )
        elif isinstance(x, TupleValue):
            return TupleValue(tuple(x.types()), tuple(x.values()))
        elif isinstance(x, ListValue):
            return TupleValue(tuple(x.types()), tuple(x.values()))
        raise TypeInferenceError(dbg, f"`tuple` operator is not defined on {x.type()}")
