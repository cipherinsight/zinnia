from typing import List, Dict, Optional

from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.debug.exception import TypeInferenceError
from zinnia.op_def.abstract.abstract_op import AbstractOp
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import Value, NDArrayValue, TupleValue, IntegerValue, FloatValue, ListValue


class PoseidonHashOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "poseidon_hash"

    @classmethod
    def get_name(cls) -> str:
        return "poseidon_hash"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("x"),
        ]

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        x = kwargs["x"]
        if isinstance(x, IntegerValue):
            return builder.ir_poseidon_hash([x])
        elif isinstance(x, FloatValue):
            raise TypeInferenceError(dbg, f"Cannot perform Poseidon hash on Float type.")
        elif isinstance(x, NDArrayValue):
            values = x.flattened_values()
            return builder.ir_poseidon_hash(values)
        elif isinstance(x, TupleValue) or isinstance(x, ListValue):
            values = x.values()
            hashes = [builder.op_poseidon_hash(v) for v in values]
            return builder.ir_poseidon_hash(hashes)
        raise TypeInferenceError(dbg, f"Operator `{self.get_name()}` on type `{x.type()}` is not defined")
