from typing import Optional, List, Dict

from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.debug.dbg_info import DebugInfo
from zinnia.debug.exception import TypeInferenceError
from zinnia.op_def.abstract.abstract_op import AbstractOp
from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import Value, NDArrayValue, ListValue, TupleValue


class AllOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "all"

    @classmethod
    def get_name(cls) -> str:
        return "all"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("x")
        ]

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        x = kwargs["x"]
        if isinstance(x, NDArrayValue):
            result = builder.ir_constant_bool(True)
            for i in range(x.shape()[0]):
                result = builder.ir_logical_and(result, builder.op_bool_cast(builder.op_ndarray_get_item(x, builder.op_square_brackets([builder.ir_constant_int(i)]))))
            return result
        elif isinstance(x, ListValue):
            result = builder.ir_constant_bool(True)
            for v in x.values():
                result = builder.ir_logical_and(result, builder.op_bool_cast(v))
            return result
        elif isinstance(x, TupleValue):
            result = builder.ir_constant_bool(True)
            for v in x.values():
                result = builder.ir_logical_and(result, builder.op_bool_cast(v))
            return result
        raise TypeInferenceError(dbg, f"Operator `{self.get_name()}` on type `{x.type()}` is not defined")
