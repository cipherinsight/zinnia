from typing import List, Dict, Any, Optional

from pyzk.opdef.nocls.abstract_op import AbstractOp, _ParamEntry
from pyzk.util.dt_descriptor import DTDescriptor, NumberDTDescriptor, NDArrayDTDescriptor
from pyzk.util.flatten_descriptor import FlattenDescriptor, NumberFlattenDescriptor, NDArrayFlattenDescriptor
from pyzk.util.inference_descriptor import InferenceDescriptor, NumberInferenceDescriptor, NDArrayInferenceDescriptor
from pyzk.util.source_pos_info import SourcePosInfo


class USubOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "usub"

    @classmethod
    def get_name(cls) -> str:
        return "usub"

    def get_param_entries(self) -> List[_ParamEntry]:
        return [
            _ParamEntry("x")
        ]

    def perform_inference(self, lhs: Any, rhs: Any) -> Any:
        raise NotImplementedError()

    def type_check(self, spi: Optional[SourcePosInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        x = kwargs["x"].type()
        if isinstance(x, NumberDTDescriptor):
            return NumberDTDescriptor()
        elif isinstance(x, NDArrayDTDescriptor):
            return NDArrayDTDescriptor(x.shape)
        raise NotImplementedError()

    def static_infer(self, spi: Optional[SourcePosInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        x = kwargs["x"]
        if isinstance(x, NumberInferenceDescriptor):
            if x.get() is None:
                return NumberInferenceDescriptor(None)
            return NumberInferenceDescriptor(x.get() * -1)
        elif isinstance(x, NDArrayInferenceDescriptor):
            return NDArrayInferenceDescriptor(x.shape(), x.get().unary(lambda a: -a if a is not None else None))
        raise NotImplementedError()

    def ir_flatten(self, ir_builder, kwargs: Dict[str, FlattenDescriptor]) -> FlattenDescriptor:
        x = kwargs["x"]
        minus_one = ir_builder.create_constant(-1)
        if isinstance(x, NumberFlattenDescriptor):
            return NumberFlattenDescriptor(ir_builder.create_mul(
                minus_one, x.ptr()
            ))
        elif isinstance(x, NDArrayFlattenDescriptor):
            return NDArrayFlattenDescriptor(x.shape(), x.ptr().unary(lambda a: ir_builder.create_mul(
                minus_one, a
            )))
        raise NotImplementedError()
