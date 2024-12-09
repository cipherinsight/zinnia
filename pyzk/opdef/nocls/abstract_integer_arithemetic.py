from typing import Callable, Any, Optional, Dict, List

from pyzk.algo.ndarray_helper import NDArrayHelper
from pyzk.debug.dbg_info import DebugInfo
from pyzk.debug.exception import TypeInferenceError
from pyzk.internal.dt_descriptor import DTDescriptor, NumberDTDescriptor, NDArrayDTDescriptor, IntegerDTDescriptor
from pyzk.internal.flatten_descriptor import FlattenDescriptor, IntegerFlattenDescriptor, NDArrayFlattenDescriptor
from pyzk.internal.inference_descriptor import InferenceDescriptor, IntegerInferenceDescriptor, \
    NDArrayInferenceDescriptor

from pyzk.opdef.nocls.abstract_op import AbstractOp


class AbstractIntegerArithemetic(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("lhs"),
            AbstractOp._ParamEntry("rhs"),
        ]

    def type_check(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        lhs, rhs = kwargs["lhs"].type(), kwargs["rhs"].type()
        if isinstance(lhs, IntegerDTDescriptor) and isinstance(rhs, IntegerDTDescriptor):
            return IntegerDTDescriptor()
        elif isinstance(lhs, IntegerDTDescriptor) and isinstance(rhs, NDArrayDTDescriptor):
            if not isinstance(rhs.dtype, IntegerDTDescriptor):
                raise TypeInferenceError(dbg_i, f'The dtype of NDArray should be `Integer` in {self.get_name()}')
            return NDArrayDTDescriptor(shape=rhs.shape, dtype=IntegerDTDescriptor())
        elif isinstance(lhs, NDArrayDTDescriptor) and isinstance(rhs, NumberDTDescriptor):
            if not isinstance(lhs.dtype, IntegerDTDescriptor):
                raise TypeInferenceError(dbg_i, f'The dtype of NDArray should be `Integer` in {self.get_name()}')
            return NDArrayDTDescriptor(shape=lhs.shape, dtype=IntegerDTDescriptor())
        elif isinstance(lhs, NDArrayDTDescriptor) and isinstance(rhs, NDArrayDTDescriptor):
            if not NDArrayHelper.broadcast_compatible(lhs.shape, rhs.shape):
                raise TypeInferenceError(dbg_i, f'Invalid binary operator `{self.get_name()}` on operands {lhs} and {rhs}, as their shapes must be broadcast compatible')
            return NDArrayDTDescriptor(
                shape=NDArrayHelper.broadcast_shape(lhs.shape, rhs.shape),
                dtype=IntegerDTDescriptor()
            )
        raise NotImplementedError()

    def static_infer(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        lhs, rhs = kwargs["lhs"], kwargs["rhs"]
        if isinstance(lhs, IntegerInferenceDescriptor) and isinstance(rhs, IntegerInferenceDescriptor):
            inference_value = self.get_inference_op_lambda()(lhs.get(), rhs.get())
            return IntegerInferenceDescriptor(inference_value)
        elif isinstance(lhs, IntegerInferenceDescriptor) and isinstance(rhs, NDArrayInferenceDescriptor):
            val = lhs.get()
            ndarray = rhs.get().unary(lambda x: self.get_inference_op_lambda()(val, x))
            return NDArrayInferenceDescriptor(rhs.shape(), IntegerDTDescriptor(), ndarray)
        elif isinstance(lhs, NDArrayInferenceDescriptor) and isinstance(rhs, IntegerInferenceDescriptor):
            val = rhs.get()
            ndarray = lhs.get().unary(lambda x: self.get_inference_op_lambda()(val, x))
            return NDArrayInferenceDescriptor(lhs.shape(), IntegerDTDescriptor(), ndarray)
        elif isinstance(lhs, NDArrayInferenceDescriptor) and isinstance(rhs, NDArrayInferenceDescriptor):
            broadcast_shape = NDArrayHelper.broadcast_shape(lhs.shape(), rhs.shape())
            _lhs, _rhs = NDArrayHelper.broadcast(lhs.get(), rhs.get())
            return NDArrayInferenceDescriptor(
                broadcast_shape,
                IntegerDTDescriptor(),
                _lhs.binary(_rhs, self.get_inference_op_lambda())
            )
        raise NotImplementedError()

    def ir_flatten(self, ir_builder, kwargs: Dict[str, FlattenDescriptor]) -> FlattenDescriptor:
        lhs, rhs = kwargs["lhs"], kwargs["rhs"]
        if isinstance(lhs, IntegerFlattenDescriptor) and isinstance(rhs, IntegerFlattenDescriptor):
            ptr = self.get_flatten_op_lambda(ir_builder)(lhs.ptr(), rhs.ptr())
            return IntegerFlattenDescriptor(ptr)
        elif isinstance(lhs, IntegerFlattenDescriptor) and isinstance(rhs, NDArrayFlattenDescriptor):
            assert NDArrayHelper.broadcast_compatible((1, ), rhs.ptr().shape)
            lhs_ndarray, rhs_ndarray = NDArrayHelper.broadcast(NDArrayHelper((1, ), [lhs.ptr()]), rhs.ptr())
            result = lhs_ndarray.binary(rhs_ndarray, self.get_flatten_op_lambda(ir_builder))
            return NDArrayFlattenDescriptor(result.shape, rhs.dtype(), result)
        elif isinstance(lhs, NDArrayFlattenDescriptor) and isinstance(rhs, IntegerFlattenDescriptor):
            assert NDArrayHelper.broadcast_compatible(lhs.ptr().shape, (1, ))
            lhs_ndarray, rhs_ndarray = NDArrayHelper.broadcast(lhs.ptr(), NDArrayHelper((1, ), [rhs.ptr()]))
            result = lhs_ndarray.binary(rhs_ndarray, self.get_flatten_op_lambda(ir_builder))
            return NDArrayFlattenDescriptor(result.shape, lhs.dtype(), result)
        elif isinstance(lhs, NDArrayFlattenDescriptor) and isinstance(rhs, NDArrayFlattenDescriptor):
            assert NDArrayHelper.broadcast_compatible(lhs.ptr().shape, rhs.ptr().shape)
            lhs_ndarray, rhs_ndarray = NDArrayHelper.broadcast(lhs.ptr(), rhs.ptr())
            result = lhs_ndarray.binary(rhs_ndarray, self.get_flatten_op_lambda(ir_builder))
            return NDArrayFlattenDescriptor(result.shape, IntegerDTDescriptor(), result)
        raise NotImplementedError()

    def get_inference_op_lambda(self) -> Callable[[Any, Any], Any]:
        raise NotImplementedError()

    def get_flatten_op_lambda(self, ir_builder) -> Callable[[int, int], int]:
        raise NotImplementedError()
