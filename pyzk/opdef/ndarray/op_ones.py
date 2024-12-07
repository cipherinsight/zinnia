from typing import List, Dict, Optional

from pyzk.debug.exception import TypeInferenceError, StaticInferenceError
from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.internal.dt_descriptor import DTDescriptor, NDArrayDTDescriptor, TupleDTDescriptor
from pyzk.internal.flatten_descriptor import FlattenDescriptor, NDArrayFlattenDescriptor
from pyzk.internal.inference_descriptor import InferenceDescriptor, NDArrayInferenceDescriptor
from pyzk.algo.ndarray_helper import NDArrayHelper
from pyzk.debug.dbg_info import DebugInfo


class NDArray_OnesOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "NDArray::ones"

    @classmethod
    def get_name(cls) -> str:
        return "NDArray::ones"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("shape")
        ]

    def type_check(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        shape = kwargs["shape"]
        if not isinstance(shape.type(), TupleDTDescriptor):
            raise TypeInferenceError(dbg_i, "Param `shape` must be of type `Tuple`")
        for ele in shape.get():
            if ele is None:
                raise StaticInferenceError(dbg_i, "Every number element in `shape` must be statically inferrable")
            if ele <= 0:
                raise TypeInferenceError(dbg_i, "Every number element in `shape` must be greater than 0")
        return NDArrayDTDescriptor(shape.get())

    def static_infer(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        shape = kwargs["shape"].get()
        ndarray = NDArrayHelper.fill(shape, lambda: 1)
        return NDArrayInferenceDescriptor(shape, ndarray)

    def ir_flatten(self, ir_builder, kwargs: Dict[str, FlattenDescriptor]) -> FlattenDescriptor:
        shape = kwargs["shape"]
        constant_1 = ir_builder.create_constant(1)
        ndarray = NDArrayHelper.fill(shape.val(), lambda: constant_1)
        return NDArrayFlattenDescriptor(shape.val(), ndarray)
