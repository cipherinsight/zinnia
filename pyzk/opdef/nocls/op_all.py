from typing import Optional, List, Dict

from pyzk.debug.dbg_info import DebugInfo
from pyzk.debug.exception import TypeInferenceError
from pyzk.internal.compiler_config import CompilerConfig
from pyzk.internal.dt_descriptor import IntegerDTDescriptor, IntegerType
from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.builder.abstract_ir_builder import AbsIRBuilderInterface
from pyzk.builder.value import Value, NDArrayValue, ListValue, TupleValue


class AllOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "all"

    @classmethod
    def get_name(cls) -> str:
        return "all"

    @classmethod
    def ir_reducible(cls, config: CompilerConfig) -> bool:
        return True

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("x")
        ]

    def build(self, reducer: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        x = kwargs["x"]
        if isinstance(x, NDArrayValue):
            result = reducer.ir_constant_int(1)
            for v in x.flattened_values():
                result = reducer.ir_logical_and(result, reducer.op_bool_scalar(v))
            return result
        elif isinstance(x, ListValue):
            result = reducer.ir_constant_int(1)
            for v in x.values():
                result = reducer.ir_logical_and(result, reducer.op_bool_scalar(v))
            return result
        elif isinstance(x, TupleValue):
            result = reducer.ir_constant_int(1)
            for v in x.values():
                result = reducer.ir_logical_and(result, reducer.op_bool_scalar(v))
            return result
        raise TypeInferenceError(dbg, f"Operator `{self.get_name()}` on type `{type(x.type())}` is not defined")
