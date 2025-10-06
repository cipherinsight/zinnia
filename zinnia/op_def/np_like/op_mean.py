from typing import Dict, Optional, List

from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.compile.triplet import Value, NumberValue, ListValue, NDArrayValue, TupleValue, NoneValue
from zinnia.debug.dbg_info import DebugInfo
from zinnia.debug.exception import TypeInferenceError
from zinnia.op_def.abstract.abstract_op import AbstractOp


class NP_MeanOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "np.mean"

    @classmethod
    def get_name(cls) -> str:
        return "mean"

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
        no_elements = 1
        for _ in a.shape():
            no_elements *= _
        return builder.op_divide(builder.op_ndarray_sum(a, axis, dbg), builder.ir_constant_int(no_elements if isinstance(axis, NoneValue) else a.shape()[axis.val()]), dbg)
