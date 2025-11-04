import copy
from typing import List, Dict, Optional

from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.debug.exception import TypeInferenceError, StaticInferenceError
from zinnia.op_def.abstract.abstract_op import AbstractOp
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import Value, NDArrayValue, IntegerValue, NumberValue, ListValue, TupleValue, \
    NoneValue


class NP_RepeatOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "np.repeat"

    @classmethod
    def get_name(cls) -> str:
        return "repeat"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("a"),
            AbstractOp._ParamEntry("repeats"),
            AbstractOp._ParamEntry("axis", True)
        ]

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        the_a = kwargs["a"]
        repeats = kwargs["repeats"]
        axis = kwargs.get("axis", builder.op_constant_none())
        if not isinstance(the_a, NDArrayValue):
            if isinstance(the_a, NumberValue):
                the_a = NDArrayValue.from_number(the_a)
            elif isinstance(the_a, ListValue) or isinstance(the_a, TupleValue):
                the_a = builder.op_np_asarray(the_a, dbg)
            else:
                raise TypeInferenceError(dbg, f"`a` must be an array, but got {the_a.type()}")
        if not isinstance(repeats, IntegerValue):
            raise TypeInferenceError(dbg, f"`repeats` must be an integer, but got {repeats.type()}")
        if isinstance(axis, NoneValue):
            axis = builder.ir_constant_int(0)
        if not isinstance(axis, IntegerValue):
            raise TypeInferenceError(dbg, f"`axis` must be an integer, but got {axis.type()}")
        if axis.val(builder) is None:
            raise StaticInferenceError(dbg, f"`axis` must be statically inferrable")
        axis_val = axis.val(builder) if axis.val(builder) >= 0 else axis.val(builder) + len(the_a.shape())
        if axis_val < 0 or axis_val >= len(the_a.shape()):
            raise StaticInferenceError(dbg, f"`axis` {axis.val(builder)} is out of bounds for array of dimension {len(the_a.shape())}")
        return NDArrayValue.concatenate(the_a.dtype(), axis.val(builder), [copy.deepcopy(the_a) for _ in range(repeats.val(builder))])
