from typing import List, Dict, Optional

from pyzk.opdef.abstract_op import AbstractOp, _ParamEntry
from pyzk.util.dt_descriptor import DTDescriptor, NumberDTDescriptor
from pyzk.util.flatten_descriptor import FlattenDescriptor, NumberFlattenDescriptor
from pyzk.util.inference_descriptor import InferenceDescriptor, NumberInferenceDescriptor
from pyzk.util.source_pos_info import SourcePosInfo


class ConstantOp(AbstractOp):
    def __init__(self, value: int):
        super().__init__()
        assert value is not None
        self.value = value

    def get_signature(self) -> str:
        return f"constant[{self.value}]"

    @classmethod
    def get_name(cls) -> str:
        return "constant"

    def get_param_entries(self) -> List[_ParamEntry]:
        return []

    def type_check(self, spi: Optional[SourcePosInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        return NumberDTDescriptor()

    def static_infer(self, spi: Optional[SourcePosInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        return NumberInferenceDescriptor(self.value)

    def ir_flatten(self, ir_builder, kwargs: Dict[str, FlattenDescriptor]) -> FlattenDescriptor:
        return NumberFlattenDescriptor(ir_builder.create_constant(self.value))
