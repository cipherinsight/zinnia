import copy
from typing import List, Dict, Optional

from zinnia.debug.exception import TypeInferenceError, StaticInferenceError
from zinnia.opdef.abstract.abstract_op import AbstractOp
from zinnia.compile.type_sys import IntegerType
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.builder.abstract_ir_builder import AbsIRBuilderInterface
from zinnia.compile.builder.value import NDArrayValue, Value, TupleValue, ListValue


class NDArray_TransposeOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "NDArray.transpose"

    @classmethod
    def get_name(cls) -> str:
        return "transpose"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("self"),
            AbstractOp._ParamEntry("axes", True),
        ]

    def build(self, builder: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        the_self = kwargs["self"]
        axes = kwargs.get("axes", None)
        assert isinstance(the_self, NDArrayValue)
        if axes is None:
            axes_vals = tuple(range(len(the_self.shape()))[::-1])
        elif isinstance(axes, TupleValue) or isinstance(axes, ListValue):
            for ele_type, ele_val in zip(axes.types(), axes.values()):
                if ele_type != IntegerType:
                    raise StaticInferenceError(dbg, f"Each element in `axes` should be an integer")
                if ele_val.val() is None:
                    raise StaticInferenceError(dbg, f"Each element in `axes` should be able to be statically inferrable")
            axes_vals = tuple(x.val() for x in axes.values())
        else:
            raise TypeInferenceError(dbg, f"`axes` should be a tuple or list of integers")
        axes_vals = tuple((ax + len(the_self.shape()) if ax < 0 else ax) for ax in axes_vals)
        if len(axes_vals) != len(the_self.shape()):
            raise TypeInferenceError(dbg, f"Length of `axes` should be equal to the number of dimensions of the array")
        permutation = [_ for _ in range(len(the_self.shape()))]
        for ax in axes_vals:
            if not 0 <= ax < len(the_self.shape()):
                raise TypeInferenceError(dbg, f"Invalid axis value `{ax}`")
            if permutation[ax] is None:
                raise TypeInferenceError(dbg, f"`axes` should be a permutation of 0 to {len(the_self.shape()) - 1}")
            permutation[ax] = None
        return copy.deepcopy(the_self.transpose(axes_vals))
