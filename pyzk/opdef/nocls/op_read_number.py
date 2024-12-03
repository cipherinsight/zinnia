from typing import List, Dict, Optional

from pyzk.opdef.abstract_op import AbstractOp, _ParamEntry
from pyzk.util.dt_descriptor import DTDescriptor, NumberDTDescriptor
from pyzk.util.inference_descriptor import InferenceDescriptor, NumberInferenceDescriptor
from pyzk.util.source_pos_info import SourcePosInfo


class ReadNumberOp(AbstractOp):
    def __init__(self, major: int, minor: int):
        super().__init__()
        self.major = major
        self.minor = minor

    def get_signature(self) -> str:
        return "read_number"

    @classmethod
    def get_name(cls) -> str:
        return "read_number"

    def get_param_entries(self) -> List[_ParamEntry]:
        return []

    def type_check(self, spi: Optional[SourcePosInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        return NumberDTDescriptor()

    def static_infer(self, spi: Optional[SourcePosInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        return NumberInferenceDescriptor(None)
