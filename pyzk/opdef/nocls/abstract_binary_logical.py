from typing import List, Dict, Any, Optional

from pyzk.debug.exception import TypeInferenceError
from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.internal.dt_descriptor import DTDescriptor, NumberDTDescriptor
from pyzk.internal.flatten_descriptor import FlattenDescriptor, NumberFlattenDescriptor
from pyzk.internal.inference_descriptor import InferenceDescriptor, NumberInferenceDescriptor
from pyzk.debug.dbg_info import DebugInfo


class AbstractBinaryLogical(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        raise NotImplementedError()

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("lhs"),
            AbstractOp._ParamEntry("rhs"),
        ]

    def perform_inference(self, lhs: Any, rhs: Any) -> Any:
        raise NotImplementedError()

    def perform_flatten(self, ir_builder, lhs: Any, rhs: Any) -> Any:
        raise NotImplementedError()

    def type_check(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        lhs, rhs = kwargs["lhs"].type(), kwargs["rhs"].type()
        if isinstance(lhs, NumberDTDescriptor) and isinstance(rhs, NumberDTDescriptor):
            return NumberDTDescriptor()
        raise TypeInferenceError(dbg_i, f'Invalid binary logical operator `{self.get_signature()}` on operands {lhs} and {rhs}, as they must be boolean values')

    def static_infer(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        lhs, rhs = kwargs["lhs"], kwargs["rhs"]
        if isinstance(lhs, NumberInferenceDescriptor) and isinstance(rhs, NumberInferenceDescriptor):
            if lhs.get() is None or rhs.get() is None:
                return NumberInferenceDescriptor(None)
            return NumberInferenceDescriptor(self.perform_inference(lhs.get(), rhs.get()))
        raise NotImplementedError()

    def ir_flatten(self, ir_builder, kwargs: Dict[str, FlattenDescriptor]) -> FlattenDescriptor:
        lhs, rhs = kwargs["lhs"], kwargs["rhs"]
        if isinstance(lhs, NumberFlattenDescriptor) and isinstance(rhs, NumberFlattenDescriptor):
            return NumberFlattenDescriptor(self.perform_flatten(ir_builder, lhs.ptr(), rhs.ptr()))
        raise NotImplementedError()
