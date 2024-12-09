from typing import List, Dict, Optional

from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.internal.dt_descriptor import DTDescriptor, NoneDTDescriptor, ClassDTDescriptor
from pyzk.internal.flatten_descriptor import FlattenDescriptor, NoneFlattenDescriptor, ClassFlattenDescriptor
from pyzk.internal.inference_descriptor import InferenceDescriptor, NoneInferenceDescriptor, ClassInferenceDescriptor
from pyzk.debug.dbg_info import DebugInfo


class ConstantClassOp(AbstractOp):
    def __init__(self, dt: DTDescriptor):
        super().__init__()
        self.dt = dt

    def get_signature(self) -> str:
        return f"constant_class"

    @classmethod
    def get_name(cls) -> str:
        return "constant_class"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return []

    def type_check(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        return ClassDTDescriptor()

    def static_infer(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        return ClassInferenceDescriptor(self.dt)

    def ir_flatten(self, ir_builder, kwargs: Dict[str, FlattenDescriptor]) -> FlattenDescriptor:
        return ClassFlattenDescriptor()
