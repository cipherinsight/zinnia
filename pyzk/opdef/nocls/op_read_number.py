from typing import List, Dict, Optional

from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.util.dt_descriptor import DTDescriptor, NumberDTDescriptor
from pyzk.util.flatten_descriptor import FlattenDescriptor, NumberFlattenDescriptor
from pyzk.util.inference_descriptor import InferenceDescriptor, NumberInferenceDescriptor
from pyzk.util.source_pos_info import SourcePosInfo


class ReadNumberOp(AbstractOp):
    def __init__(self, major: int, minor: int):
        super().__init__()
        self.major = major
        self.minor = minor

    def get_signature(self) -> str:
        return f"read_number[{self.major}, {self.minor}]"

    @classmethod
    def get_name(cls) -> str:
        return "read_number"

    def dce_keep(self) -> bool:
        return True

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return []

    def type_check(self, spi: Optional[SourcePosInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        return NumberDTDescriptor()

    def static_infer(self, spi: Optional[SourcePosInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        return NumberInferenceDescriptor(None)

    def ir_flatten(self, ir_builder, kwargs: Dict[str, FlattenDescriptor]) -> FlattenDescriptor:
        return NumberFlattenDescriptor(ir_builder.create_read_number(self.major, self.minor))
