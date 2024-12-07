from typing import List, Dict, Optional

from pyzk.debug.exception import TypeInferenceError
from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.internal.dt_descriptor import DTDescriptor, NDArrayDTDescriptor, TupleDTDescriptor
from pyzk.internal.flatten_descriptor import FlattenDescriptor, TupleFlattenDescriptor
from pyzk.internal.inference_descriptor import InferenceDescriptor, TupleInferenceDescriptor
from pyzk.debug.dbg_info import DebugInfo


class NDArray_ShapeOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "NDArray::shape"

    @classmethod
    def get_name(cls) -> str:
        return "NDArray::shape"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("self")
        ]

    def type_check(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        the_self = kwargs["self"]
        if not isinstance(the_self.type(), NDArrayDTDescriptor):
            raise TypeInferenceError(dbg_i, "Param `self` must be of type `NDArray`")
        return TupleDTDescriptor(len(the_self.shape()))

    def static_infer(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        the_self = kwargs["self"]
        shape = the_self.shape()
        return TupleInferenceDescriptor(len(shape), shape)

    def ir_flatten(self, ir_builder, kwargs: Dict[str, FlattenDescriptor]) -> FlattenDescriptor:
        the_self = kwargs["self"]
        shape = the_self.shape()
        return TupleFlattenDescriptor(len(shape), tuple(ir_builder.create_constant(x) for x in shape))
