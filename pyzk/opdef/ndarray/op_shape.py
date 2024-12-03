from typing import List, Dict, Optional

from pyzk.exception.contextual import TypeInferenceError
from pyzk.opdef.abstract_op import AbstractOp, _ParamEntry
from pyzk.util.dt_descriptor import DTDescriptor, NDArrayDTDescriptor, TupleDTDescriptor
from pyzk.util.inference_descriptor import InferenceDescriptor, TupleInferenceDescriptor
from pyzk.util.source_pos_info import SourcePosInfo


class NDArray_ShapeOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "NDArray::shape"

    @classmethod
    def get_name(cls) -> str:
        return "NDArray::shape"

    def get_param_entries(self) -> List[_ParamEntry]:
        return [
            _ParamEntry("self")
        ]

    def type_check(self, spi: Optional[SourcePosInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        the_self = kwargs["self"]
        if not isinstance(the_self.type(), NDArrayDTDescriptor):
            raise TypeInferenceError(spi, "Param `self` must be of type `NDArray`")
        return TupleDTDescriptor(len(the_self.shape()))

    def static_infer(self, spi: Optional[SourcePosInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        the_self = kwargs["self"]
        shape = the_self.shape()
        return TupleInferenceDescriptor(len(shape), shape)
