from typing import List, Dict, Optional, Any

from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.debug.exception import TypeInferenceError
from zinnia.compile.type_sys import NDArrayDTDescriptor
from zinnia.op_def.abstract.abstract_op import AbstractOp
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import Value, NDArrayValue, TupleValue, ListValue


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

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        if "x" not in kwargs:
            return TupleValue(tuple(), tuple())
        x = kwargs["x"]
        if isinstance(x, NDArrayValue):
            sub_element_shape = x.shape()[1:]
            sub_element_type = NDArrayDTDescriptor(sub_element_shape, x.dtype())
            return TupleValue(
                tuple(sub_element_type for _ in range(x.shape()[0])),
                tuple(builder.op_get_item(x, builder.op_square_brackets([builder.ir_constant_int(i)])) for i in range(x.shape()[0]))
            )
        elif isinstance(x, TupleValue):
            return TupleValue(tuple(x.types()), tuple(x.values()))
        elif isinstance(x, ListValue):
            return TupleValue(tuple(x.types()), tuple(x.values()))
        raise TypeInferenceError(dbg, f"`tuple` operator is not defined on {x.type()}")
