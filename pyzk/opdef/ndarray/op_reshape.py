from typing import List, Dict, Optional

from pyzk.debug.exception import TypeInferenceError, StaticInferenceError
from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.internal.dt_descriptor import DTDescriptor, NDArrayDTDescriptor
from pyzk.internal.flatten_descriptor import FlattenDescriptor, NDArrayFlattenDescriptor
from pyzk.internal.inference_descriptor import InferenceDescriptor, NDArrayInferenceDescriptor, \
    NDArrayInferenceValue, TupleInferenceDescriptor
from pyzk.debug.dbg_info import DebugInfo


class NDArray_ReshapeOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "NDArray::reshape"

    @classmethod
    def get_name(cls) -> str:
        return "NDArray::reshape"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("self"),
            AbstractOp._ParamEntry("shape"),
        ]

    def type_check(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        the_self = kwargs["self"]
        the_shape = kwargs["shape"]
        if not isinstance(the_self, NDArrayInferenceDescriptor):
            raise TypeInferenceError(dbg_i, f"`{self.get_name()}` can only be used on `NDArray`")
        if not isinstance(the_shape, TupleInferenceDescriptor):
            raise TypeInferenceError(dbg_i, f"`shape` of `{self.get_name()}` must be a Tuple")
        else:
            num_elements = 1
            for element in the_shape.get():
                if element is None:
                    raise StaticInferenceError(dbg_i, f"Cannot statically infer the value of the argument `shape`")
                num_elements *= element
            num_elements_self = 1
            for element in the_self.shape():
                num_elements_self *= element
            if num_elements != num_elements_self:
                raise TypeInferenceError(dbg_i, f"Number of elements in `shape` must be equal to the number of elements in the original `NDArray`")
        return NDArrayDTDescriptor(the_shape.get(), the_self.dtype())

    def static_infer(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        the_self = kwargs["self"]
        the_shape = kwargs["shape"]
        flattened_values = the_self.get().flatten()
        new_values = NDArrayInferenceValue.from_1d_values_and_shape(flattened_values, the_shape.get())
        return NDArrayInferenceDescriptor(new_values.shape, the_self.dtype(), new_values)

    def ir_flatten(self, ir_builder, kwargs: Dict[str, FlattenDescriptor]) -> FlattenDescriptor:
        the_self = kwargs["self"]
        the_shape = kwargs["shape"]
        flattened_values = the_self.ptr().flatten()
        new_values = NDArrayInferenceValue.from_1d_values_and_shape(flattened_values, the_shape.val())
        return NDArrayFlattenDescriptor(new_values.shape, the_self.dtype(), new_values)
