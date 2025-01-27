from typing import Optional, List, Dict

from zinnia.debug.dbg_info import DebugInfo
from zinnia.debug.exception import TypeInferenceError
from zinnia.opdef.nocls.abstract_op import AbstractOp
from zinnia.compile.builder.abstract_ir_builder import AbsIRBuilderInterface
from zinnia.compile.builder.value import Value, NDArrayValue, ListValue, TupleValue


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

    def build(self, builder: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        x = kwargs["x"]
        if isinstance(x, NDArrayValue):
            result = builder.ir_constant_int(1)
            for v in x.flattened_values():
                result = builder.ir_logical_and(result, builder.op_bool_scalar(v))
            return result
        elif isinstance(x, ListValue):
            result = builder.ir_constant_int(1)
            for v in x.values():
                result = builder.ir_logical_and(result, builder.op_bool_scalar(v))
            return result
        elif isinstance(x, TupleValue):
            result = builder.ir_constant_int(1)
            for v in x.values():
                result = builder.ir_logical_and(result, builder.op_bool_scalar(v))
            return builder
        raise TypeInferenceError(dbg, f"Operator `{self.get_name()}` on type `{x.type()}` is not defined")
