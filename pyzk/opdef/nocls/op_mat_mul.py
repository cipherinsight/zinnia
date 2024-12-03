from typing import List, Dict, Optional

from pyzk.exception.contextual import TypeInferenceError
from pyzk.opdef.abstract_op import AbstractOp, _ParamEntry
from pyzk.util.dt_descriptor import DTDescriptor, NDArrayDTDescriptor
from pyzk.util.inference_descriptor import NDArrayInferenceDescriptor, InferenceDescriptor
from pyzk.util.ndarray_helper import NDArrayHelper
from pyzk.util.source_pos_info import SourcePosInfo


class MatMulOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "mat_mul"

    @classmethod
    def get_name(cls) -> str:
        return "mat_mul"

    def get_param_entries(self) -> List[_ParamEntry]:
        return [
            _ParamEntry("lhs"),
            _ParamEntry("rhs"),
        ]

    def type_check(self, spi: Optional[SourcePosInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        lhs, rhs = kwargs["lhs"].type(), kwargs["rhs"].type()
        if isinstance(lhs, NDArrayDTDescriptor) and isinstance(rhs, NDArrayDTDescriptor):
            if not NDArrayHelper.matmul_shape_matches(lhs.shape, rhs.shape):
                raise TypeInferenceError(spi, f'Invalid binary operator `{self.get_signature()}` on operands {lhs} and {rhs}, as their shapes are not multiply compatible')
            return NDArrayDTDescriptor(shape=NDArrayHelper.matmul_shape(lhs.shape, rhs.shape))
        raise TypeInferenceError(spi, f'Invalid binary operator `{self.get_signature()}` on operands {lhs} and {rhs}. Only ndarray can be passed to `{self.get_signature()}`')

    def static_infer(self, spi: Optional[SourcePosInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        lhs, rhs = kwargs["lhs"], kwargs["rhs"]
        if isinstance(lhs, NDArrayInferenceDescriptor) and isinstance(rhs, NDArrayInferenceDescriptor):
            matmul_shape = NDArrayHelper.matmul_shape(lhs.shape(), rhs.shape())
            return NDArrayInferenceDescriptor(matmul_shape, NDArrayHelper.matmul(
                lhs.get(), rhs.get(), lambda x, y: x + y if x is not None and y is not None else None, lambda x, y: x * y if x is not None and y is not None else None, lambda: 0))
        raise NotImplementedError
