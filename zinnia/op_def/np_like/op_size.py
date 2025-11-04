from typing import List, Dict, Optional

from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.debug.exception import TypeInferenceError, StaticInferenceError
from zinnia.op_def.abstract.abstract_op import AbstractOp
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import Value, NDArrayValue, NumberValue, ListValue, TupleValue, NoneValue, \
    IntegerValue


class NP_SizeOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "np.size"

    @classmethod
    def get_name(cls) -> str:
        return "size"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("self"),
            AbstractOp._ParamEntry("axis", True)
        ]

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        the_self = kwargs["self"]
        the_axis = kwargs.get("axis", builder.op_constant_none())
        if isinstance(the_self, NumberValue):
            the_self = NDArrayValue.from_number(the_self)
        if isinstance(the_self, ListValue) or isinstance(the_self, TupleValue):
            the_self = builder.op_np_asarray(the_self, dbg)
        if not isinstance(the_self, NDArrayValue):
            raise TypeInferenceError(dbg, f"Expected NDArray, got {the_self.type()}")
        if isinstance(the_axis, NoneValue):
            number_elements = 1
            for i in the_self.shape():
                number_elements *= i
            return builder.ir_constant_int(number_elements)
        if not isinstance(the_axis, IntegerValue):
            raise TypeInferenceError(dbg, f"Expected `axis` to be an integer, got {the_axis.type()}")
        if the_axis.val(builder) is None:
            raise StaticInferenceError(dbg, f"Cannot statically infer the value to `axis`")
        axis_val = the_axis.val(builder) if the_axis.val(builder) >= 0 else len(the_self.shape()) + the_axis.val(builder)
        if axis_val < 0 or axis_val >= len(the_self.shape()):
            raise StaticInferenceError(dbg, f"`axis` {the_axis.val(builder)} is out of bounds for array of dimension {len(the_self.shape())}")
        return builder.ir_constant_int(the_self.shape()[axis_val])
