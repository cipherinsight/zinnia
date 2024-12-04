from typing import List, Dict, Optional

from pyzk.exception.contextual import TypeInferenceError, StaticInferenceError
from pyzk.opdef.nocls.abstract_op import AbstractOp, _ParamEntry
from pyzk.util.dt_descriptor import DTDescriptor, NumberDTDescriptor, NoneDTDescriptor
from pyzk.util.flatten_descriptor import FlattenDescriptor, NoneFlattenDescriptor
from pyzk.util.inference_descriptor import InferenceDescriptor, NoneInferenceDescriptor
from pyzk.util.source_pos_info import SourcePosInfo


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

    def get_param_entries(self) -> List[_ParamEntry]:
        return [
            _ParamEntry("test")
        ]

    def type_check(self, spi: Optional[SourcePosInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        operand = kwargs["test"].type()
        if isinstance(operand, NumberDTDescriptor):
            return NoneDTDescriptor()
        raise TypeInferenceError(spi, f"Type `{operand}` is not supported on operator `{self.get_signature()}`")

    def static_infer(self, spi: Optional[SourcePosInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        operand = kwargs["test"]
        if operand.get() is not None and operand.get() == 0:
            raise StaticInferenceError(spi, "Assertion is always unsatisfiable")
        return NoneInferenceDescriptor()

    def ir_flatten(self, ir_builder, kwargs: Dict[str, FlattenDescriptor]) -> FlattenDescriptor:
        operand = kwargs["test"]
        ir_builder.create_assert(operand.ptr())
        return NoneFlattenDescriptor()
