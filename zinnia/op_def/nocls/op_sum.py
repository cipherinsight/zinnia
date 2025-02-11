import copy
from typing import List, Dict, Optional

from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.debug.dbg_info import DebugInfo
from zinnia.debug.exception import TypeInferenceError
from zinnia.op_def.abstract.abstract_op import AbstractOp
from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import Value, NDArrayValue, ListValue, TupleValue, NumberValue, NoneValue


class SumOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "sum"

    @classmethod
    def get_name(cls) -> str:
        return "sum"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("iterable"),
            AbstractOp._ParamEntry("start", True)
        ]

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        iterable = kwargs["iterable"]
        start = kwargs.get("start", builder.op_constant_none())
        if isinstance(iterable, NDArrayValue):
            result = builder.op_get_item(iterable, builder.op_square_brackets([builder.ir_constant_int(0)]), dbg)
            for i in range(1, iterable.shape()[0]):
                result = builder.op_add(result, builder.op_get_item(iterable, builder.op_square_brackets([builder.ir_constant_int(i)]), dbg), dbg)
            if not isinstance(start, NoneValue):
                return builder.op_add(result, start, dbg)
            return result
        elif isinstance(iterable, ListValue) or isinstance(iterable, TupleValue):
            result = iterable.values()[0]
            for val in iterable.values()[1:]:
                result = builder.op_add(result, val, dbg)
            if not isinstance(start, NoneValue):
                return builder.op_add(result, start, dbg)
            return result
        elif isinstance(iterable, NumberValue):
            if not isinstance(start, NoneValue):
                return builder.op_add(iterable, start, dbg)
            return copy.copy(iterable)
        raise TypeInferenceError(dbg, f"`sum` not defined on {iterable.type()}")
