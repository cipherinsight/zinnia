from typing import List, Dict, Optional

from pyzk.debug.exception import TypeInferenceError
from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.internal.dt_descriptor import DTDescriptor, NumberDTDescriptor, NDArrayDTDescriptor, TupleDTDescriptor
from pyzk.internal.flatten_descriptor import FlattenDescriptor, NDArrayFlattenDescriptor, TupleFlattenDescriptor, \
    NumberFlattenDescriptor
from pyzk.internal.inference_descriptor import InferenceDescriptor, NumberInferenceDescriptor, NDArrayInferenceDescriptor, \
    TupleInferenceDescriptor
from pyzk.debug.dbg_info import DebugInfo


class LenOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "len"

    @classmethod
    def get_name(cls) -> str:
        return "len"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("operand")
        ]

    def type_check(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        operand = kwargs["operand"].type()
        if isinstance(operand, NDArrayDTDescriptor):
            return NumberDTDescriptor()
        elif isinstance(operand, TupleDTDescriptor):
            return NumberDTDescriptor()
        raise TypeInferenceError(dbg_i, f"Type `{operand}` is not supported on operator `{self.get_signature()}`")

    def static_infer(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
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
