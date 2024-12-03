from typing import List, Dict, Tuple, Optional

from pyzk.opdef.abstract_op import AbstractOp, _ParamEntry
from pyzk.util.datatype_name import DataTypeName
from pyzk.util.dt_descriptor import DTDescriptor, NumberDTDescriptor, NDArrayDTDescriptor
from pyzk.util.inference_descriptor import InferenceDescriptor, NumberInferenceDescriptor, NDArrayInferenceDescriptor
from pyzk.util.ndarray_helper import NDArrayHelper
from pyzk.util.source_pos_info import SourcePosInfo


class InputOp(AbstractOp):
    def __init__(self, input_id: int, typename: str, shape: Tuple[int, ...], public: bool):
        super().__init__()
        self.input_id = input_id
        self.typename = typename
        self.shape = shape
        self.public = public

    def get_signature(self) -> str:
        return "input"

    @classmethod
    def get_name(cls) -> str:
        return "input"

    def get_param_entries(self) -> List[_ParamEntry]:
        return []

    def type_check(self, spi: Optional[SourcePosInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        if self.typename == DataTypeName.NDARRAY:
            return NDArrayDTDescriptor(self.shape)
        elif self.typename == DataTypeName.NUMBER:
            return NumberDTDescriptor()
        raise NotImplementedError()

    def static_infer(self, spi: Optional[SourcePosInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        if self.typename == DataTypeName.NDARRAY:
            return NDArrayInferenceDescriptor(self.shape, NDArrayHelper.fill(self.shape, lambda: None))
        elif self.typename == DataTypeName.NUMBER:
            return NumberInferenceDescriptor(None)
        raise NotImplementedError()
