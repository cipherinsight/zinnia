from typing import List, Dict, Optional

from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.internal.dt_descriptor import DTDescriptor, FloatDTDescriptor
from pyzk.internal.flatten_descriptor import FlattenDescriptor, FloatFlattenDescriptor
from pyzk.internal.inference_descriptor import InferenceDescriptor, FloatInferenceDescriptor
from pyzk.debug.dbg_info import DebugInfo


class ConstantFloatOp(AbstractOp):
    def __init__(self, value: int):
        super().__init__()
        assert value is not None
        self.value = value

    def get_signature(self) -> str:
        return f"constant_float[{self.value}]"

    @classmethod
    def get_name(cls) -> str:
        return "constant_float"

    def __eq__(self, other):
        return super().__eq__(other) and self.value == other.value

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return []

    def type_check(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        return FloatDTDescriptor()

    def static_infer(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        return FloatInferenceDescriptor(self.value)

    def ir_flatten(self, ir_builder, kwargs: Dict[str, FlattenDescriptor]) -> FlattenDescriptor:
        return FloatFlattenDescriptor(ir_builder.create_constant_float(self.value))
