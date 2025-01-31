from typing import Optional, List, Dict

from zinnia.debug.dbg_info import DebugInfo
from zinnia.debug.exception import TypeInferenceError
from zinnia.opdef.abstract.abstract_op import AbstractOp
from zinnia.compile.builder.abstract_ir_builder import AbsIRBuilderInterface
from zinnia.compile.builder.value import NDArrayValue, Value, ListValue, TupleValue


class AnyOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "any"

    @classmethod
    def get_name(cls) -> str:
        return "any"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("x")
        ]

    def build(self, builder: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        x = kwargs["x"]
        if isinstance(x, NDArrayValue):
            result = builder.ir_constant_int(0)
            for i in range(x.shape()[0]):
                result = builder.ir_logical_or(result, builder.op_bool_cast(builder.op_ndarray_get_item(x, builder.op_square_brackets([builder.ir_constant_int(i)]))))
            return result
        elif isinstance(x, ListValue):
            result = builder.ir_constant_int(0)
            for v in x.values():
                result = builder.ir_logical_or(result, builder.op_bool_cast(v))
            return result
        elif isinstance(x, TupleValue):
            result = builder.ir_constant_int(0)
            for v in x.values():
                result = builder.ir_logical_or(result, builder.op_bool_cast(v))
            return result
        raise TypeInferenceError(dbg, f"Operator `{self.get_name()}` on type `{x.type()}` is not defined")
