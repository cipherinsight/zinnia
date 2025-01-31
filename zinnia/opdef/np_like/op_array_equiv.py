from typing import Dict, Optional, List

from zinnia.compile.builder.abstract_ir_builder import AbsIRBuilderInterface
from zinnia.compile.builder.value import Value, NDArrayValue, ListValue, NumberValue, TupleValue
from zinnia.debug.dbg_info import DebugInfo
from zinnia.opdef.abstract.abstract_op import AbstractOp


class NP_ArrayEquivOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "np.array_equiv"

    @classmethod
    def get_name(cls) -> str:
        return "array_equiv"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("x1"),
            AbstractOp._ParamEntry("x2"),
        ]

    def build(self, builder: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        x1, x2 = kwargs["x1"], kwargs["x2"]
        if isinstance(x1, NumberValue) and isinstance(x2, NumberValue):
            return builder.op_equal(x1, x2, dbg)
        if isinstance(x1, ListValue) or isinstance(x1, TupleValue):
            x1 = builder.op_ndarray_asarray(x1, dbg)
        if isinstance(x2, ListValue) or isinstance(x2, TupleValue):
            x2 = builder.op_ndarray_asarray(x2, dbg)
        if isinstance(x1, NumberValue):
            x1 = NDArrayValue.from_number(x1)
        if isinstance(x2, NumberValue):
            x2 = NDArrayValue.from_number(x2)
        if not isinstance(x1, NDArrayValue) or not isinstance(x2, NDArrayValue):
            return builder.ir_constant_int(0)
        return builder.op_ndarray_all(builder.op_equal(x1, x2, dbg), dbg)
