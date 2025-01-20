from typing import List, Dict, Optional

from zenopy.debug.exception import TypeInferenceError
from zenopy.opdef.nocls.abstract_op import AbstractOp
from zenopy.debug.dbg_info import DebugInfo
from zenopy.builder.abstract_ir_builder import AbsIRBuilderInterface
from zenopy.builder.value import Value, NDArrayValue, TupleValue, IntegerValue, FloatValue, ListValue


class HashOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "hash"

    @classmethod
    def get_name(cls) -> str:
        return "hash"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("x"),
        ]

    def build(self, reducer: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        x = kwargs["x"]
        if isinstance(x, IntegerValue):
            return reducer.ir_hash([x])
        elif isinstance(x, FloatValue):
            return reducer.ir_hash([x])
        elif isinstance(x, NDArrayValue):
            values = x.flattened_values()
            return reducer.ir_hash(values)
        elif isinstance(x, TupleValue) or isinstance(x, ListValue):
            values = x.values()
            hashes = [reducer.op_hash(v) for v in values]
            return reducer.ir_hash(hashes)
        raise TypeInferenceError(dbg, f"Operator `{self.get_name()}` on type `{x.type()}` is not defined")
