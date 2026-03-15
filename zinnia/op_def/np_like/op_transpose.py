from typing import List, Optional

from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.compile.triplet import ListValue, NDArrayValue, NoneValue, NumberValue, TupleValue, Value
from zinnia.debug.dbg_info import DebugInfo
from zinnia.debug.exception import TypeInferenceError
from zinnia.op_def.abstract.abstract_op import AbstractOp


class NP_TransposeOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "np.transpose"

    @classmethod
    def get_name(cls) -> str:
        return "transpose"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("a"),
            AbstractOp._ParamEntry("axes", True),
        ]

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        a = kwargs["a"]
        axes = kwargs.get("axes", builder.op_constant_none())
        if isinstance(a, NumberValue):
            a = NDArrayValue.from_number(a)
        if isinstance(a, ListValue) or isinstance(a, TupleValue):
            a = builder.op_np_asarray(a, dbg)
        if not isinstance(a, NDArrayValue):
            raise TypeInferenceError(dbg, f"Operator `{self.get_name()}` on type `{a.type()}` is not defined. `a` must be a NDArray.")
        if not isinstance(axes, (TupleValue, ListValue, NoneValue)):
            raise TypeInferenceError(dbg, f"`axes` should be a tuple or list of integers, but got {axes.type()}")
        return builder.op_ndarray_transpose(a, axes, dbg)