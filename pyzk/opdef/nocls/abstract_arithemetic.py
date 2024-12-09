from typing import List, Dict, Callable, Any, Optional

from pyzk.debug.exception import TypeInferenceError
from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.internal.dt_descriptor import DTDescriptor, IntegerDTDescriptor, NDArrayDTDescriptor, FloatDTDescriptor, \
    NumberDTDescriptor
from pyzk.internal.flatten_descriptor import FlattenDescriptor, IntegerFlattenDescriptor, NDArrayFlattenDescriptor, \
    FloatFlattenDescriptor, NumberFlattenDescriptor
from pyzk.internal.inference_descriptor import InferenceDescriptor, IntegerInferenceDescriptor, \
    NDArrayInferenceDescriptor, FloatInferenceDescriptor, NumberInferenceDescriptor
from pyzk.algo.ndarray_helper import NDArrayHelper
from pyzk.debug.dbg_info import DebugInfo


class AbstractArithemetic(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        raise NotImplementedError()

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("lhs"),
            AbstractOp._ParamEntry("rhs"),
        ]

    def get_inference_op_lambda(self, lhs_dt: DTDescriptor, rhs_dt: DTDescriptor) -> Callable[[Any, Any], Any]:
        raise NotImplementedError()

    def get_flatten_op_lambda(self, ir_builder, lhs_dt: DTDescriptor, rhs_dt: DTDescriptor) -> Callable[[int, int], int]:
        raise NotImplementedError()

    def get_expected_result_dt(self, lhs_dt: DTDescriptor, rhs_dt: DTDescriptor):
        if isinstance(lhs_dt, FloatDTDescriptor) and isinstance(rhs_dt, FloatDTDescriptor):
            return FloatDTDescriptor()
        elif isinstance(lhs_dt, IntegerDTDescriptor) and isinstance(rhs_dt, FloatDTDescriptor):
            return FloatDTDescriptor()
        elif isinstance(lhs_dt, FloatDTDescriptor) and isinstance(rhs_dt, IntegerDTDescriptor):
            return FloatDTDescriptor()
        elif isinstance(lhs_dt, IntegerDTDescriptor) and isinstance(rhs_dt, IntegerDTDescriptor):
            return IntegerDTDescriptor()
        raise NotImplementedError()

    def type_check(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        lhs, rhs = kwargs["lhs"].type(), kwargs["rhs"].type()
        if isinstance(lhs, NumberDTDescriptor) and isinstance(rhs, NumberDTDescriptor):
            return self.get_expected_result_dt(lhs, rhs)
        elif isinstance(lhs, NumberDTDescriptor) and isinstance(rhs, NDArrayDTDescriptor):
            return NDArrayDTDescriptor(shape=rhs.shape, dtype=self.get_expected_result_dt(lhs, rhs.dtype))
        elif isinstance(lhs, NDArrayDTDescriptor) and isinstance(rhs, NumberDTDescriptor):
            return NDArrayDTDescriptor(shape=lhs.shape, dtype=self.get_expected_result_dt(lhs.dtype, rhs))
        elif isinstance(lhs, NDArrayDTDescriptor) and isinstance(rhs, NDArrayDTDescriptor):
            if not NDArrayHelper.broadcast_compatible(lhs.shape, rhs.shape):
                raise TypeInferenceError(dbg_i, f'Invalid binary operator `{self.get_signature()}` on operands {lhs} and {rhs}, as their shapes must be broadcast compatible')
            return NDArrayDTDescriptor(
                shape=NDArrayHelper.broadcast_shape(lhs.shape, rhs.shape),
                dtype=self.get_expected_result_dt(lhs.dtype, rhs.dtype)
            )
        raise NotImplementedError()

    def static_infer(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        lhs, rhs = kwargs["lhs"], kwargs["rhs"]
        if isinstance(lhs, NumberInferenceDescriptor) and isinstance(rhs, NumberInferenceDescriptor):
            inference_value = self.get_inference_op_lambda(lhs.dt, rhs.dt)(lhs.get(), rhs.get())
            expected_dt = self.get_expected_result_dt(lhs.dt, rhs.dt)
            if isinstance(expected_dt, FloatDTDescriptor):
                return FloatInferenceDescriptor(inference_value)
            elif isinstance(expected_dt, IntegerDTDescriptor):
                return IntegerInferenceDescriptor(inference_value)
            raise NotImplementedError()
        elif isinstance(lhs, NumberInferenceDescriptor) and isinstance(rhs, NDArrayInferenceDescriptor):
            val = lhs.get()
            ndarray = rhs.get().unary(lambda x: self.get_inference_op_lambda(lhs.dt, rhs.dtype())(val, x))
            return NDArrayInferenceDescriptor(rhs.shape(), self.get_expected_result_dt(lhs.dt, rhs.dtype()), ndarray)
        elif isinstance(lhs, NDArrayInferenceDescriptor) and isinstance(rhs, NumberInferenceDescriptor):
            val = rhs.get()
            ndarray = lhs.get().unary(lambda x: self.get_inference_op_lambda(lhs.dtype(), rhs.dt)(val, x))
            return NDArrayInferenceDescriptor(lhs.shape(), self.get_expected_result_dt(lhs.dtype(), rhs.dt), ndarray)
        elif isinstance(lhs, NDArrayInferenceDescriptor) and isinstance(rhs, NDArrayInferenceDescriptor):
            broadcast_shape = NDArrayHelper.broadcast_shape(lhs.shape(), rhs.shape())
            _lhs, _rhs = NDArrayHelper.broadcast(lhs.get(), rhs.get())
            return NDArrayInferenceDescriptor(
                broadcast_shape,
                self.get_expected_result_dt(lhs.dtype(), rhs.dtype()),
                _lhs.binary(_rhs, self.get_inference_op_lambda(lhs.dtype(), rhs.dtype()))
            )
        raise NotImplementedError()

    def ir_flatten(self, ir_builder, kwargs: Dict[str, FlattenDescriptor]) -> FlattenDescriptor:
        lhs, rhs = kwargs["lhs"], kwargs["rhs"]
        if isinstance(lhs, NumberFlattenDescriptor) and isinstance(rhs, NumberFlattenDescriptor):
            ptr = self.get_flatten_op_lambda(ir_builder, lhs.dt, rhs.dt)(lhs.ptr(), rhs.ptr())
            expected_dt = self.get_expected_result_dt(lhs.dt, rhs.dt)
            if isinstance(expected_dt, FloatDTDescriptor):
                return FloatFlattenDescriptor(ptr)
            elif isinstance(expected_dt, IntegerDTDescriptor):
                return IntegerFlattenDescriptor(ptr)
            raise NotImplementedError()
        elif isinstance(lhs, NumberFlattenDescriptor) and isinstance(rhs, NDArrayFlattenDescriptor):
            assert NDArrayHelper.broadcast_compatible((1, ), rhs.ptr().shape)
            lhs_ndarray, rhs_ndarray = NDArrayHelper.broadcast(NDArrayHelper((1, ), [lhs.ptr()]), rhs.ptr())
            result = lhs_ndarray.binary(rhs_ndarray, self.get_flatten_op_lambda(ir_builder, lhs.dt, rhs.dtype()))
            return NDArrayFlattenDescriptor(result.shape, rhs.dtype(), result)
        elif isinstance(lhs, NDArrayFlattenDescriptor) and isinstance(rhs, NumberFlattenDescriptor):
            assert NDArrayHelper.broadcast_compatible(lhs.ptr().shape, (1, ))
            lhs_ndarray, rhs_ndarray = NDArrayHelper.broadcast(lhs.ptr(), NDArrayHelper((1, ), [rhs.ptr()]))
            result = lhs_ndarray.binary(rhs_ndarray, self.get_flatten_op_lambda(ir_builder, lhs.dtype(), rhs.dt))
            return NDArrayFlattenDescriptor(result.shape, lhs.dtype(), result)
        elif isinstance(lhs, NDArrayFlattenDescriptor) and isinstance(rhs, NDArrayFlattenDescriptor):
            assert NDArrayHelper.broadcast_compatible(lhs.ptr().shape, rhs.ptr().shape)
            lhs_ndarray, rhs_ndarray = NDArrayHelper.broadcast(lhs.ptr(), rhs.ptr())
            result = lhs_ndarray.binary(rhs_ndarray, self.get_flatten_op_lambda(ir_builder, lhs.dtype(), rhs.dtype()))
            return NDArrayFlattenDescriptor(result.shape, self.get_expected_result_dt(lhs.dtype(), rhs.dtype()), result)
        raise NotImplementedError()
