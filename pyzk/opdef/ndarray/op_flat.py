from typing import List, Dict, Optional

from pyzk.debug.exception import TypeInferenceError
from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.internal.dt_descriptor import DTDescriptor, NDArrayDTDescriptor
from pyzk.internal.flatten_descriptor import FlattenDescriptor, TupleFlattenDescriptor, NDArrayFlattenDescriptor
from pyzk.internal.inference_descriptor import InferenceDescriptor, TupleInferenceDescriptor, \
    NDArrayInferenceDescriptor, NDArrayInferenceValue
from pyzk.debug.dbg_info import DebugInfo


class NDArray_FlatOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "NDArray::flat"

    @classmethod
    def get_name(cls) -> str:
        return "NDArray::flat"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("self")
        ]

    def type_check(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        the_self = kwargs["self"]
        if not isinstance(the_self, NDArrayInferenceDescriptor):
            raise TypeInferenceError(dbg_i, "Param `self` must be of type `NDArray`")
        the_shape = the_self.shape()
        num_items = 1
        for x in the_shape:
            num_items *= x
        return NDArrayDTDescriptor((num_items, ), the_self.dtype())

    def static_infer(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        the_self = kwargs["self"]
        assert isinstance(the_self, NDArrayInferenceDescriptor)
        the_shape = the_self.shape()
        num_items = 1
        for x in the_shape:
            num_items *= x
        flatten_items = the_self.get().flatten()
        return NDArrayInferenceDescriptor((num_items, ), the_self.dtype(), NDArrayInferenceValue((num_items, ), flatten_items))

    def ir_flatten(self, ir_builder, kwargs: Dict[str, FlattenDescriptor]) -> FlattenDescriptor:
        the_self = kwargs["self"]
        assert isinstance(the_self, NDArrayFlattenDescriptor)
        the_shape = the_self.shape()
        num_items = 1
        for x in the_shape:
            num_items *= x
        flatten_items = the_self.ptr().flatten()
        return NDArrayFlattenDescriptor((num_items, ), the_self.dtype(), NDArrayInferenceValue((num_items, ), flatten_items))
