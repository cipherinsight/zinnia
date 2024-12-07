from typing import List, Dict, Any, Optional

from pyzk.debug.exception import TypeInferenceError
from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.internal.dt_descriptor import DTDescriptor, NumberDTDescriptor
from pyzk.internal.flatten_descriptor import FlattenDescriptor, NumberFlattenDescriptor
from pyzk.internal.inference_descriptor import InferenceDescriptor, NumberInferenceDescriptor
from pyzk.debug.dbg_info import DebugInfo


class BoolCastOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "bool_cast"

    @classmethod
    def get_name(cls) -> str:
        return "bool_cast"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("x")
        ]

    def perform_inference(self, lhs: Any, rhs: Any) -> Any:
        raise NotImplementedError()

    def type_check(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        x = kwargs["x"].type()
        if isinstance(x, NumberDTDescriptor):
            return NumberDTDescriptor()
        raise TypeInferenceError(dbg_i, f'Invalid logical operator `{self.get_signature()}` on operand {x}, as it must be a number')

    def static_infer(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        x = kwargs["x"]
        if isinstance(x, NumberInferenceDescriptor):
            if x.get() is None:
                return NumberInferenceDescriptor(None)
            return NumberInferenceDescriptor(1 if x != 0 else 0)
        raise NotImplementedError()

    def ir_flatten(self, ir_builder, kwargs: Dict[str, FlattenDescriptor]) -> FlattenDescriptor:
        x = kwargs["x"]
        return NumberFlattenDescriptor(ir_builder.create_not_equal(x.ptr(), ir_builder.create_constant(0)))
