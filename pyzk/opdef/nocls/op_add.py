from typing import Callable, Any, Dict, Optional

from pyzk.opdef.abstract_arithemetic import AbstractArithemetic
from pyzk.util.dt_descriptor import DTDescriptor, TupleDTDescriptor
from pyzk.util.inference_descriptor import InferenceDescriptor, TupleInferenceDescriptor
from pyzk.util.source_pos_info import SourcePosInfo


class AddOp(AbstractArithemetic):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "add"

    @classmethod
    def get_name(cls) -> str:
        return "add"

    def get_inference_op_lambda(self) -> Callable[[Any, Any], Any]:
        return lambda x, y: x + y if x is not None and y is not None else None

    def type_check(self, spi: Optional[SourcePosInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        lhs, rhs = kwargs["lhs"].type(), kwargs["rhs"].type()
        if isinstance(lhs, TupleDTDescriptor) and isinstance(rhs, TupleDTDescriptor):
            return TupleDTDescriptor(lhs.length + rhs.length)
        return super().type_check(spi, kwargs)

    def static_infer(self, spi: Optional[SourcePosInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        lhs, rhs = kwargs["lhs"], kwargs["rhs"]
        if isinstance(lhs, TupleInferenceDescriptor) and isinstance(rhs, TupleInferenceDescriptor):
            return TupleInferenceDescriptor(lhs.length() + rhs.length(), lhs.get() + rhs.get())
        return super().static_infer(spi, kwargs)
