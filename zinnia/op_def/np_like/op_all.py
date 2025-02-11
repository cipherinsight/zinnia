from typing import Dict, Optional, List

from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.compile.triplet import Value, NDArrayValue, NumberValue, ListValue, TupleValue
from zinnia.debug.dbg_info import DebugInfo
from zinnia.debug.exception import TypeInferenceError
from zinnia.op_def.abstract.abstract_op import AbstractOp


class NP_AllOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "np.all"

    @classmethod
    def get_name(cls) -> str:
        return "all"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("a"),
            AbstractOp._ParamEntry("axis", True),
        ]

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        a, axis = kwargs["a"], kwargs.get("axis", builder.op_constant_none())
        if isinstance(a, NumberValue):
            a = NDArrayValue.from_number(a)
        if isinstance(a, ListValue) or isinstance(a, TupleValue):
            a = builder.op_np_asarray(a, dbg)
        if not isinstance(a, NDArrayValue):
            raise TypeInferenceError(dbg, f"Operator `{self.get_name()}` on type `{a.type()}` is not defined. `a` must be a NDArray.")
        return builder.op_ndarray_all(a, axis, dbg)
