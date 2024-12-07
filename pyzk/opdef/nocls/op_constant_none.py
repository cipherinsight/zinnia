from typing import List, Dict, Optional

from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.internal.dt_descriptor import DTDescriptor, NoneDTDescriptor
from pyzk.internal.flatten_descriptor import FlattenDescriptor, NoneFlattenDescriptor
from pyzk.internal.inference_descriptor import InferenceDescriptor, NoneInferenceDescriptor
from pyzk.debug.dbg_info import DebugInfo


class ConstantNoneOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return f"constant_none"

    @classmethod
    def get_name(cls) -> str:
        return "constant_none"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return []

    def type_check(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        return NoneDTDescriptor()

    def static_infer(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        return NoneInferenceDescriptor()

    def ir_flatten(self, ir_builder, kwargs: Dict[str, FlattenDescriptor]) -> FlattenDescriptor:
        return NoneFlattenDescriptor()
