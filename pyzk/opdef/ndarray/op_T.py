from typing import List, Dict, Optional

from pyzk.debug.exception import TypeInferenceError
from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.internal.dt_descriptor import DTDescriptor, NDArrayDTDescriptor
from pyzk.internal.flatten_descriptor import FlattenDescriptor, NDArrayFlattenDescriptor
from pyzk.internal.inference_descriptor import InferenceDescriptor, NDArrayInferenceDescriptor, \
    NDArrayInferenceValue
from pyzk.debug.dbg_info import DebugInfo


class NDArray_TOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "NDArray::T"

    @classmethod
    def get_name(cls) -> str:
        return "NDArray::T"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("self"),
        ]

    def type_check(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        the_self = kwargs["self"]
        if not isinstance(the_self, NDArrayInferenceDescriptor):
            raise TypeInferenceError(dbg_i, f"`{self.get_name()}` can only be used on `NDArray`")
        return NDArrayDTDescriptor(the_self.shape()[::-1], the_self.shape())

    def static_infer(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        the_self = kwargs["self"]
        dtype = the_self.shape()
        new_shape = the_self.shape()[::-1]
        flattened_values = the_self.get().flatten()
        return NDArrayInferenceDescriptor(new_shape, dtype, NDArrayInferenceValue.from_1d_values_and_shape(flattened_values, new_shape))

    def ir_flatten(self, ir_builder, kwargs: Dict[str, FlattenDescriptor]) -> FlattenDescriptor:
        the_self = kwargs["self"]
        new_shape = the_self.shape()[::-1]
        flattened_values = the_self.ptr().flatten()
        return NDArrayFlattenDescriptor(new_shape, dtype, NDArrayInferenceValue.from_1d_values_and_shape(flattened_values, new_shape))
