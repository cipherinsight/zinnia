from typing import List, Dict, Optional

from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.debug.exception import TypeInferenceError
from zinnia.op_def.abstract.abstract_op import AbstractOp
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import Value, NDArrayValue, NumberValue, TupleValue, ListValue


class NP_AppendOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "NDArray.append"

    @classmethod
    def get_name(cls) -> str:
        return "append"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("arr"),
            AbstractOp._ParamEntry("values"),
            AbstractOp._ParamEntry("axis", True)
        ]

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        the_arr = kwargs["arr"]
        values = kwargs["values"]
        axis = kwargs.get("axis", builder.op_constant_none())
        if isinstance(the_arr, NumberValue):
            the_arr = NDArrayValue.from_number(the_arr)
        if isinstance(the_arr, TupleValue) or isinstance(the_arr, ListValue):
            the_arr = builder.op_np_asarray(the_arr, dbg)
        if isinstance(values, NumberValue):
            values = NDArrayValue.from_number(values)
        if isinstance(values, TupleValue) or isinstance(values, ListValue):
            values = builder.op_np_asarray(values, dbg)
        if not isinstance(the_arr, NDArrayValue):
            raise TypeInferenceError(dbg, f"`arr` must be an NDArray, but got {the_arr.type()}")
        if not isinstance(values, NDArrayValue):
            raise TypeInferenceError(dbg, f"`values` must be an NDArray, but got {values.type()}")
        return builder.op_np_concatenate(builder.op_parenthesis([the_arr, values]), axis, dbg)
