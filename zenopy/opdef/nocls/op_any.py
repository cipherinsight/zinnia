from typing import Optional, List, Dict

from zenopy.debug.dbg_info import DebugInfo
from zenopy.debug.exception import TypeInferenceError
from zenopy.opdef.nocls.abstract_op import AbstractOp
from zenopy.builder.abstract_ir_builder import AbsIRBuilderInterface
from zenopy.builder.value import NDArrayValue, Value, ListValue, TupleValue


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

    def build(self, reducer: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        x = kwargs["x"]
        if isinstance(x, NDArrayValue):
            result = reducer.ir_constant_int(0)
            for v in x.flattened_values():
                result = reducer.ir_logical_or(result, reducer.op_bool_scalar(v))
            return result
        elif isinstance(x, ListValue):
            result = reducer.ir_constant_int(0)
            for v in x.values():
                result = reducer.ir_logical_or(result, reducer.op_bool_scalar(v))
            return result
        elif isinstance(x, TupleValue):
            result = reducer.ir_constant_int(0)
            for v in x.values():
                result = reducer.ir_logical_or(result, reducer.op_bool_scalar(v))
            return result
        raise TypeInferenceError(dbg, f"Operator `{self.get_name()}` on type `{x.type()}` is not defined")
