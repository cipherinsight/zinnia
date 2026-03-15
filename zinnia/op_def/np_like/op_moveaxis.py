from typing import List, Optional

from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.compile.triplet import IntegerValue, ListValue, NDArrayValue, NumberValue, TupleValue, Value
from zinnia.debug.dbg_info import DebugInfo
from zinnia.debug.exception import TypeInferenceError
from zinnia.op_def.abstract.abstract_op import AbstractOp


class NP_MoveAxisOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "np.moveaxis"

    @classmethod
    def get_name(cls) -> str:
        return "moveaxis"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("a"),
            AbstractOp._ParamEntry("source"),
            AbstractOp._ParamEntry("destination"),
        ]

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        a = kwargs["a"]
        source = kwargs["source"]
        destination = kwargs["destination"]
        if isinstance(a, NumberValue):
            a = NDArrayValue.from_number(a)
        if isinstance(a, ListValue) or isinstance(a, TupleValue):
            a = builder.op_np_asarray(a, dbg)
        if not isinstance(a, NDArrayValue):
            raise TypeInferenceError(dbg, f"Operator `{self.get_name()}` on type `{a.type()}` is not defined. `a` must be a NDArray.")
        if not isinstance(source, IntegerValue):
            raise TypeInferenceError(dbg, f"`source` must be an integer, but got {source.type()}")
        if not isinstance(destination, IntegerValue):
            raise TypeInferenceError(dbg, f"`destination` must be an integer, but got {destination.type()}")
        return builder.op_ndarray_moveaxis(a, source, destination, dbg)