from typing import Dict, Optional, List

from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.compile.triplet import Value, NDArrayValue, NumberValue, ListValue, TupleValue
from zinnia.debug.dbg_info import DebugInfo
from zinnia.op_def.abstract.abstract_op import AbstractOp


class NP_ArrayEqualOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "np.array_equal"

    @classmethod
    def get_name(cls) -> str:
        return "array_equal"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("x1"),
            AbstractOp._ParamEntry("x2"),
        ]

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        x1, x2 = kwargs["x1"], kwargs["x2"]
        if isinstance(x1, NumberValue) and isinstance(x2, NumberValue):
            return builder.op_equal(x1, x2, dbg)
        if isinstance(x1, ListValue) or isinstance(x1, TupleValue):
            x1 = builder.op_np_asarray(x1, dbg)
        if isinstance(x2, ListValue) or isinstance(x2, TupleValue):
            x2 = builder.op_np_asarray(x2, dbg)
        if not isinstance(x1, NDArrayValue) or not isinstance(x2, NDArrayValue):
            return builder.ir_constant_bool(False)
        if x1.shape() != x2.shape():
            return builder.ir_constant_bool(False)
        return builder.op_ndarray_all(builder.op_equal(x1, x2, dbg), dbg)
