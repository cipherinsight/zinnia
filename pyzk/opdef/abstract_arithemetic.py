from typing import List, Dict, Callable, Any, Optional

from pyzk.exception.contextual import TypeInferenceError
from pyzk.opdef.abstract_op import AbstractOp, _ParamEntry
from pyzk.util.dt_descriptor import DTDescriptor, NumberDTDescriptor, NDArrayDTDescriptor
from pyzk.util.inference_descriptor import InferenceDescriptor, NumberInferenceDescriptor, NDArrayInferenceDescriptor
from pyzk.util.ndarray_helper import NDArrayHelper
from pyzk.util.source_pos_info import SourcePosInfo


class AbstractArithemetic(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        raise NotImplementedError()

    def get_param_entries(self) -> List[_ParamEntry]:
        return [
            _ParamEntry("lhs"),
            _ParamEntry("rhs"),
        ]

    def get_inference_op_lambda(self) -> Callable[[Any, Any], Any]:
        raise NotImplementedError()

    def type_check(self, spi: Optional[SourcePosInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        lhs, rhs = kwargs["lhs"].type(), kwargs["rhs"].type()
        if isinstance(lhs, NumberDTDescriptor) and isinstance(rhs, NumberDTDescriptor):
            return NumberDTDescriptor()
        elif isinstance(lhs, NumberDTDescriptor) and isinstance(rhs, NDArrayDTDescriptor):
            return NDArrayDTDescriptor(shape=rhs.shape)
        elif isinstance(lhs, NDArrayDTDescriptor) and isinstance(rhs, NumberDTDescriptor):
            return NDArrayDTDescriptor(shape=lhs.shape)
        elif isinstance(lhs, NDArrayDTDescriptor) and isinstance(rhs, NDArrayDTDescriptor):
            if not NDArrayHelper.broadcast_compatible(lhs.shape, rhs.shape):
                raise TypeInferenceError(spi, f'Invalid binary operator `{self.get_signature()}` on operands {lhs} and {rhs}, as their shapes must be broadcast compatible')
            return NDArrayDTDescriptor(shape=NDArrayHelper.broadcast_shape(lhs.shape, rhs.shape))
        raise NotImplementedError()

    def static_infer(self, spi: Optional[SourcePosInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        lhs, rhs = kwargs["lhs"], kwargs["rhs"]
        if isinstance(lhs, NumberInferenceDescriptor) and isinstance(rhs, NumberInferenceDescriptor):
            if lhs.get() is None or rhs.get() is None:
                return NumberInferenceDescriptor()
            return NumberInferenceDescriptor(self.get_inference_op_lambda()(lhs.get(), rhs.get()))
        elif isinstance(lhs, NumberInferenceDescriptor) and isinstance(rhs, NDArrayInferenceDescriptor):
            if lhs.get() is None:
                return NDArrayInferenceDescriptor.new_instance(rhs, rhs.get().unary(lambda x: None))
            ndarray = rhs.get()
            val = lhs.get()
            return NDArrayInferenceDescriptor.new_instance(rhs, ndarray.unary(lambda x: self.get_inference_op_lambda()(val, x)))
        elif isinstance(lhs, NDArrayInferenceDescriptor) and isinstance(rhs, NumberInferenceDescriptor):
            if rhs.get() is None:
                return NDArrayInferenceDescriptor.new_instance(lhs, lhs.get().unary(lambda x: None))
            ndarray = lhs.get()
            val = rhs.get()
            return NDArrayInferenceDescriptor.new_instance(lhs, ndarray.unary(lambda x: self.get_inference_op_lambda()(x, val)))
        elif isinstance(lhs, NDArrayInferenceDescriptor) and isinstance(rhs, NDArrayInferenceDescriptor):
            broadcast_shape = NDArrayHelper.broadcast_shape(lhs.shape(), rhs.shape())
            _lhs, _rhs = NDArrayHelper.broadcast(lhs.get(), rhs.get())
            return NDArrayInferenceDescriptor(broadcast_shape, _lhs.binary(_rhs, self.get_inference_op_lambda()))
        raise NotImplementedError()
