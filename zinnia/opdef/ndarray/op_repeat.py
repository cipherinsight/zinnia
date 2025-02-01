import copy
from typing import List, Dict, Optional

from zinnia.debug.exception import TypeInferenceError, StaticInferenceError
from zinnia.opdef.abstract.abstract_op import AbstractOp
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.builder.abstract_ir_builder import AbsIRBuilderInterface
from zinnia.compile.builder.value import Value, NDArrayValue, IntegerValue, NoneValue


class NDArray_RepeatOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "NDArray.repeat"

    @classmethod
    def get_name(cls) -> str:
        return "repeat"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("self"),
            AbstractOp._ParamEntry("repeats"),
            AbstractOp._ParamEntry("axis", True)
        ]

    def build(self, builder: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        the_self = kwargs["self"]
        repeats = kwargs["repeats"]
        axis = kwargs.get("axis", builder.op_constant_none())
        assert isinstance(the_self, NDArrayValue)
        if not isinstance(repeats, IntegerValue):
            raise TypeInferenceError(dbg, f"`repeats` must be an integer, but got {repeats.type()}")
        if isinstance(axis, NoneValue):
            axis = builder.ir_constant_int(0)
        if not isinstance(axis, IntegerValue):
            raise TypeInferenceError(dbg, f"`axis` must be an integer, but got {axis.type()}")
        if axis.val() is None:
            raise StaticInferenceError(dbg, f"`axis` must be statically inferrable")
        axis_val = axis.val() if axis.val() >= 0 else axis.val() + len(the_self.shape())
        if axis_val < 0 or axis_val >= len(the_self.shape()):
            raise StaticInferenceError(dbg, f"`axis` {axis.val()} is out of bounds for array of dimension {len(the_self.shape())}")
        return NDArrayValue.concatenate(the_self.dtype(), axis.val(), [copy.deepcopy(the_self) for _ in range(repeats.val())])
