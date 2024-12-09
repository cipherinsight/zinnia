from typing import List, Dict, Optional

from pyzk.debug.exception import TypeInferenceError, StaticInferenceError
from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.internal.dt_descriptor import DTDescriptor, IntegerDTDescriptor, NoneDTDescriptor
from pyzk.internal.flatten_descriptor import FlattenDescriptor, NoneFlattenDescriptor
from pyzk.internal.inference_descriptor import InferenceDescriptor, NoneInferenceDescriptor
from pyzk.debug.dbg_info import DebugInfo


class AssertOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "assert"

    @classmethod
    def get_name(cls) -> str:
        return "assert"

    def dce_keep(self) -> bool:
        return True

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("test")
        ]

    def type_check(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        operand = kwargs["test"].type()
        if isinstance(operand, IntegerDTDescriptor):
            return NoneDTDescriptor()
        raise TypeInferenceError(dbg_i, f"Type `{operand}` is not supported on operator `{self.get_signature()}`")

    def static_infer(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        operand = kwargs["test"]
        if operand.get() is not None and operand.get() == 0:
            raise StaticInferenceError(dbg_i, "Assertion is always unsatisfiable")
        return NoneInferenceDescriptor()

    def ir_flatten(self, ir_builder, kwargs: Dict[str, FlattenDescriptor]) -> FlattenDescriptor:
        operand = kwargs["test"]
        ir_builder.create_assert(operand.ptr())
        return NoneFlattenDescriptor()
