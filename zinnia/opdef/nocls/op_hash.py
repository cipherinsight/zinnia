from typing import List, Dict, Optional

from zinnia.debug.exception import TypeInferenceError
from zinnia.opdef.nocls.abstract_op import AbstractOp
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.builder.abstract_ir_builder import AbsIRBuilderInterface
from zinnia.compile.builder.value import Value, NDArrayValue, TupleValue, IntegerValue, FloatValue, ListValue


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

    def build(self, builder: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        x = kwargs["x"]
        if isinstance(x, IntegerValue):
            return builder.ir_hash([x])
        elif isinstance(x, FloatValue):
            return builder.ir_hash([x])
        elif isinstance(x, NDArrayValue):
            values = x.flattened_values()
            return builder.ir_hash(values)
        elif isinstance(x, TupleValue) or isinstance(x, ListValue):
            values = x.values()
            hashes = [builder.op_hash(v) for v in values]
            return builder.ir_hash(hashes)
        raise TypeInferenceError(dbg, f"Operator `{self.get_name()}` on type `{x.type()}` is not defined")
