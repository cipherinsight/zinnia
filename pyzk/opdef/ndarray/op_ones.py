from typing import List, Dict, Optional

from pyzk.exception.contextual import TypeInferenceError, StaticInferenceError
from pyzk.opdef.abstract_op import AbstractOp, _ParamEntry
from pyzk.util.dt_descriptor import DTDescriptor, NDArrayDTDescriptor, TupleDTDescriptor
from pyzk.util.inference_descriptor import InferenceDescriptor, NDArrayInferenceDescriptor
from pyzk.util.ndarray_helper import NDArrayHelper
from pyzk.util.source_pos_info import SourcePosInfo


class NDArray_OnesOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "NDArray::ones"

    @classmethod
    def get_name(cls) -> str:
        return "NDArray::ones"

    def get_param_entries(self) -> List[_ParamEntry]:
        return [
            _ParamEntry("shape")
        ]

    def type_check(self, spi: Optional[SourcePosInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        shape = kwargs["shape"]
        if not isinstance(shape.type(), TupleDTDescriptor):
            raise TypeInferenceError(spi, "Param `shape` must be of type `Tuple`")
        for ele in shape.get():
            if ele is None:
                raise StaticInferenceError(spi, "Every number element in `shape` must be statically inferrable")
            if ele <= 0:
                raise TypeInferenceError(spi, "Every number element in `shape` must be greater than 0")
        return NDArrayDTDescriptor(shape.get())

    def static_infer(self, spi: Optional[SourcePosInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        shape = kwargs["shape"].get()
        ndarray = NDArrayHelper.fill(shape, lambda: 1)
        return NDArrayInferenceDescriptor(shape, ndarray)
