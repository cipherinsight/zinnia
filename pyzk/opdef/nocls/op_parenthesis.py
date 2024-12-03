from typing import Dict, Any, Optional

from pyzk.exception.contextual import TypeInferenceError
from pyzk.opdef.abstract_op import AbstractOp
from pyzk.util.dt_descriptor import DTDescriptor, NumberDTDescriptor, TupleDTDescriptor
from pyzk.util.inference_descriptor import InferenceDescriptor, TupleInferenceDescriptor
from pyzk.util.source_pos_info import SourcePosInfo


class ParenthesisOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "parenthesis"

    @classmethod
    def get_name(cls) -> str:
        return "parenthesis"

    def params_parse(self, spi: Optional[SourcePosInfo], *args, **kwargs) -> Dict[str, Any]:
        if len(kwargs.items()) != 0:
            raise ValueError("Internal Error: `kwargs` Should be empty here")
        return {f'_{i}': arg for i, arg in enumerate(args)}

    def type_check(self, spi: Optional[SourcePosInfo], **kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        args = kwargs.values()
        if all([isinstance(arg, NumberDTDescriptor) for arg in args]):
            return TupleDTDescriptor(len(args))
        raise TypeInferenceError(spi,"Create Tuple using parenthesis failed: only `Number` can be accepted as elements")

    def static_infer(self, spi: Optional[SourcePosInfo], **kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        args = kwargs.values()
        return TupleInferenceDescriptor(len(args), tuple(arg.get() for arg in args))
