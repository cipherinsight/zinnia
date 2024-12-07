from typing import List, Dict, Optional

from pyzk.debug.exception import TypeInferenceError, StaticInferenceError
from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.internal.dt_descriptor import DTDescriptor, NDArrayDTDescriptor
from pyzk.internal.flatten_descriptor import FlattenDescriptor, NDArrayFlattenDescriptor, TupleFlattenDescriptor
from pyzk.internal.inference_descriptor import InferenceDescriptor, TupleInferenceDescriptor, NDArrayInferenceDescriptor, \
    NDArrayInferenceValue
from pyzk.debug.dbg_info import DebugInfo


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

    def type_check(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        the_self = kwargs["self"]
        axes = kwargs["axes"]
        if not isinstance(the_self, NDArrayInferenceDescriptor):
            raise TypeInferenceError(dbg_i, f"`{self.get_name()}` can only be used on `NDArray`")
        if axes is None:
            axes_vals = reversed([x for x in range(len(the_self.shape()))])
        elif isinstance(axes, TupleInferenceDescriptor):
            tuple_vals = axes.get()
            for tuple_val in tuple_vals:
                if tuple_val is None:
                    raise StaticInferenceError(dbg_i, f"Each element in `axes` should be able to be statically inferred")
            axes_vals = axes.get()
        elif isinstance(axes, NDArrayInferenceDescriptor):
            ndarray_vals = axes.get()
            if len(ndarray_vals.shape) != 1:
                raise TypeInferenceError(dbg_i, "Invalid provided `NDArray` on `axes`, the shape of this `NDArray` should be exactly 1-dimension")
            for ele_val in ndarray_vals.values:
                if ele_val is None:
                    raise StaticInferenceError(dbg_i, f"Each element in `axes` should be able to be statically inferred")
            axes_vals = axes.get().values
        else:
            raise TypeInferenceError(dbg_i, "Invalid type on `axes`")
        permutation = [_ for _ in range(len(the_self.shape()))]
        for axe in axes_vals:
            if not 0 <= axe < len(the_self.shape()):
                raise TypeInferenceError(dbg_i, f"Invalid axis value `{axe}`")
            if permutation[axe] is None:
                raise TypeInferenceError(dbg_i, f"`axes` should be a permutation of 0 to {len(the_self.shape()) - 1}")
            permutation[axe] = None
        return NDArrayDTDescriptor(tuple(the_self.shape()[x] for x in axes_vals))

    def static_infer(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        the_self = kwargs["self"]
        axes = kwargs["axes"]
        if axes is None:
            axes_vals = reversed([x for x in range(len(the_self.shape()))])
        elif isinstance(axes, TupleInferenceDescriptor):
            axes_vals = axes.get()
        elif isinstance(axes, NDArrayInferenceDescriptor):
            axes_vals = axes.get().values
        else:
            raise NotImplementedError()
        flattened_values = the_self.get().flatten()
        new_shape = tuple(the_self.shape()[x] for x in axes_vals)
        return NDArrayInferenceDescriptor(new_shape, NDArrayInferenceValue.from_1d_values_and_shape(flattened_values, new_shape))

    def ir_flatten(self, ir_builder, kwargs: Dict[str, FlattenDescriptor]) -> FlattenDescriptor:
        the_self = kwargs["self"]
        axes = kwargs["axes"]
        if axes is None:
            axes_vals = reversed([x for x in range(len(the_self.shape()))])
        elif isinstance(axes, TupleFlattenDescriptor):
            axes_vals = axes.val()
        elif isinstance(axes, NDArrayFlattenDescriptor):
            axes_vals = axes.val().values
        else:
            raise NotImplementedError()
        flattened_values = the_self.ptr().flatten()
        new_shape = tuple(the_self.shape()[x] for x in axes_vals)
        return NDArrayFlattenDescriptor(new_shape, NDArrayInferenceValue.from_1d_values_and_shape(flattened_values, new_shape))
