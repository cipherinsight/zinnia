from typing import List, Dict, Optional

from pyzk.exception.contextual import TypeInferenceError
from pyzk.opdef.nocls.abstract_op import AbstractOp, _ParamEntry
from pyzk.util.dt_descriptor import DTDescriptor, NumberDTDescriptor, NDArrayDTDescriptor, TupleDTDescriptor
from pyzk.util.flatten_descriptor import FlattenDescriptor, NDArrayFlattenDescriptor, TupleFlattenDescriptor, \
    NumberFlattenDescriptor
from pyzk.util.inference_descriptor import InferenceDescriptor, NumberInferenceDescriptor, NDArrayInferenceDescriptor, \
    TupleInferenceDescriptor
from pyzk.util.source_pos_info import SourcePosInfo


class LenOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "len"

    @classmethod
    def get_name(cls) -> str:
        return "len"

    def get_param_entries(self) -> List[_ParamEntry]:
        return [
            _ParamEntry("operand")
        ]

    def type_check(self, spi: Optional[SourcePosInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        operand = kwargs["operand"].type()
        if isinstance(operand, NDArrayDTDescriptor):
            return NumberDTDescriptor()
        elif isinstance(operand, TupleDTDescriptor):
            return NumberDTDescriptor()
        raise TypeInferenceError(spi, f"Type `{operand}` is not supported on operator `{self.get_signature()}`")

    def static_infer(self, spi: Optional[SourcePosInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        operand = kwargs["operand"]
        if isinstance(operand, NDArrayInferenceDescriptor):
            return NumberInferenceDescriptor(operand.shape()[0])
        elif isinstance(operand, TupleInferenceDescriptor):
            return NumberInferenceDescriptor(operand.length())
        raise NotImplementedError()

    def ir_flatten(self, ir_builder, kwargs: Dict[str, FlattenDescriptor]) -> FlattenDescriptor:
        operand = kwargs["operand"]
        if isinstance(operand, NDArrayFlattenDescriptor):
            return NumberFlattenDescriptor(ir_builder.create_constant(operand.shape()[0]))
        elif isinstance(operand, TupleFlattenDescriptor):
            return NumberFlattenDescriptor(ir_builder.create_constant(operand.length()))
        raise NotImplementedError()
