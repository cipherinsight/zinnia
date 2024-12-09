from typing import List, Dict, Optional

from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.internal.dt_descriptor import DTDescriptor, FloatDTDescriptor
from pyzk.internal.flatten_descriptor import FlattenDescriptor, FloatFlattenDescriptor
from pyzk.internal.inference_descriptor import InferenceDescriptor, FloatInferenceDescriptor
from pyzk.debug.dbg_info import DebugInfo


class ReadFloatOp(AbstractOp):
    def __init__(self, major: int, minor: int):
        super().__init__()
        self.major = major
        self.minor = minor

    def get_signature(self) -> str:
        return f"read_float[{self.major}, {self.minor}]"

    @classmethod
    def get_name(cls) -> str:
        return "read_float"

    def dce_keep(self) -> bool:
        return True

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return []

    def type_check(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        return FloatDTDescriptor()

    def static_infer(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        return FloatInferenceDescriptor(None)

    def ir_flatten(self, ir_builder, kwargs: Dict[str, FlattenDescriptor]) -> FlattenDescriptor:
        return FloatFlattenDescriptor(ir_builder.create_read_float(self.major, self.minor))
