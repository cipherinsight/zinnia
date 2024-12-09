from typing import Dict, Any, Optional, List

from pyzk.debug.exception import TypeInferenceError
from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.internal.dt_descriptor import DTDescriptor, TupleDTDescriptor
from pyzk.internal.flatten_descriptor import FlattenDescriptor, TupleFlattenDescriptor
from pyzk.internal.inference_descriptor import InferenceDescriptor, TupleInferenceDescriptor, IntegerInferenceDescriptor
from pyzk.debug.dbg_info import DebugInfo


class ParenthesisOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "parenthesis"

    @classmethod
    def get_name(cls) -> str:
        return "parenthesis"

    def params_parse(self, dbg_i: Optional[DebugInfo], args: List[InferenceDescriptor], kwargs: Dict[str, InferenceDescriptor]) -> Dict[str, Any]:
        if len(kwargs.items()) != 0:
            raise ValueError("Internal Error: `kwargs` Should be empty here")
        return {f'_{i}': arg for i, arg in enumerate(args)}

    def type_check(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        args = kwargs.values()
        if all([isinstance(arg, IntegerInferenceDescriptor) for arg in args]):
            return TupleDTDescriptor(len(args))
        raise TypeInferenceError(dbg_i,"Create Tuple using parenthesis failed: only `Integer` can be accepted as elements")

    def static_infer(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        args = kwargs.values()
        return TupleInferenceDescriptor(len(args), tuple(arg.get() for arg in args))

    def ir_flatten(self, ir_builder, kwargs: Dict[str, FlattenDescriptor]) -> FlattenDescriptor:
        args = kwargs.values()
        return TupleFlattenDescriptor(len(args), tuple(arg.ptr() for arg in args))
