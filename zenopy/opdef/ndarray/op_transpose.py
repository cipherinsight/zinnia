from typing import List, Dict, Optional

from zenopy.algo.ndarray_helper import NDArrayValueWrapper
from zenopy.debug.exception import TypeInferenceError, StaticInferenceError
from zenopy.opdef.nocls.abstract_op import AbstractOp
from zenopy.internal.dt_descriptor import IntegerType
from zenopy.debug.dbg_info import DebugInfo
from zenopy.builder.abstract_ir_builder import AbsIRBuilderInterface
from zenopy.builder.value import NDArrayValue, Value, TupleValue, ListValue


class NDArray_TransposeOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "NDArray::transpose"

    @classmethod
    def get_name(cls) -> str:
        return "NDArray::transpose"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("self"),
            AbstractOp._ParamEntry("axes"),
        ]

    def build(self, reducer: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        the_self = kwargs["self"]
        axes = kwargs["axes"]
        assert isinstance(the_self, NDArrayValue)
        if axes is None:
            axes_vals = reversed([x for x in range(len(the_self.shape()))])
        elif isinstance(axes, TupleValue) or isinstance(axes, ListValue):
            for ele_type, ele_val in zip(axes.types(), axes.values()):
                if ele_type != IntegerType:
                    raise StaticInferenceError(dbg, f"Each element in `axes` should be an integer")
                if ele_val.val() is None:
                    raise StaticInferenceError(dbg, f"Each element in `axes` should be able to be statically inferred")
            axes_vals = [x.val() for x in axes.values()]
        else:
            raise TypeInferenceError(dbg, f"`axes` should be a tuple or list of integers")
        permutation = [_ for _ in range(len(the_self.shape()))]
        for ax in axes_vals:
            if not 0 <= ax < len(the_self.shape()):
                raise TypeInferenceError(dbg, f"Invalid axis value `{ax}`")
            if permutation[ax] is None:
                raise TypeInferenceError(dbg, f"`axes` should be a permutation of 0 to {len(the_self.shape()) - 1}")
            permutation[ax] = None
        new_shape = tuple(the_self.shape()[x] for x in axes_vals)
        flattened_values = the_self.get().flatten()
        new_values = NDArrayValueWrapper.from_1d_values_and_shape(flattened_values, new_shape)
        return NDArrayValue(new_shape, the_self.dtype(), new_values)
