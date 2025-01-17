from typing import List, Dict, Callable, Optional

from pyzk.debug.exception import TypeInferenceError
from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.internal.dt_descriptor import DTDescriptor, IntegerDTDescriptor, FloatDTDescriptor, FloatType, IntegerType
from pyzk.algo.ndarray_helper import NDArrayValueWrapper
from pyzk.debug.dbg_info import DebugInfo
from pyzk.builder.abstract_ir_builder import AbsIRBuilderInterface
from pyzk.builder.value import NDArrayValue, NumberValue, Value


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

    def get_reduce_op_lambda(
            self,
            reducer: AbsIRBuilderInterface,
    ) -> Callable[[NumberValue, NumberValue], NumberValue]:
        raise NotImplementedError()

    def get_expected_result_dt(self, lhs_dt: DTDescriptor, rhs_dt: DTDescriptor):
        if lhs_dt == FloatType and rhs_dt == FloatType:
            return FloatDTDescriptor()
        elif lhs_dt == FloatType and rhs_dt == IntegerType:
            return FloatDTDescriptor()
        elif lhs_dt == IntegerType and rhs_dt == FloatType:
            return FloatDTDescriptor()
        elif lhs_dt == IntegerType and rhs_dt == IntegerType:
            return IntegerDTDescriptor()
        raise NotImplementedError()

    def reduce_number_and_number(self, reducer: AbsIRBuilderInterface, lhs: NumberValue, rhs: NumberValue) -> Value:
        return self.get_reduce_op_lambda(reducer)(lhs, rhs)

    def reduce_number_and_ndarray(self, reducer: AbsIRBuilderInterface, lhs: NumberValue, rhs: NDArrayValue) -> Value:
        lhs_ndarray = NDArrayValue.from_number(lhs)
        lhs_ndarray, rhs_ndarray = NDArrayValue.broadcast(lhs_ndarray, rhs)
        result = NDArrayValue.binary(lhs_ndarray, rhs_ndarray, self.get_expected_result_dt(lhs.type(), rhs.dtype()), self.get_reduce_op_lambda(reducer))
        return result

    def reduce_ndarray_and_number(self, reducer: AbsIRBuilderInterface, lhs: NDArrayValue, rhs: NumberValue) -> Value:
        rhs_ndarray = NDArrayValue.from_number(rhs)
        lhs_ndarray, rhs_ndarray = NDArrayValue.broadcast(lhs, rhs_ndarray)
        result = NDArrayValue.binary(lhs_ndarray, rhs_ndarray, self.get_expected_result_dt(lhs.dtype(), rhs.type()), self.get_reduce_op_lambda(reducer))
        return result

    def reduce_ndarray_and_ndarray(self, reducer: AbsIRBuilderInterface, lhs: NDArrayValue, rhs: NDArrayValue) -> NDArrayValue:
        if not NDArrayValueWrapper.binary_broadcast_compatible(lhs.shape(), rhs.shape()):
            raise TypeInferenceError(None, f"Cannot broadcast two NDArray with shapes {lhs.shape()} and {rhs.shape()}")
        lhs, rhs = NDArrayValue.broadcast(lhs, rhs)
        result = NDArrayValue.binary(lhs, rhs, self.get_expected_result_dt(lhs.dtype(), rhs.dtype()), self.get_reduce_op_lambda(reducer))
        return result

    def build(self, reducer: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        lhs, rhs = kwargs["lhs"], kwargs["rhs"]
        if isinstance(lhs, NumberValue) and isinstance(rhs, NumberValue):
            return self.reduce_number_and_number(reducer, lhs, rhs)
        elif isinstance(lhs, NumberValue) and isinstance(rhs, NDArrayValue):
            return self.reduce_number_and_ndarray(reducer, lhs, rhs)
        elif isinstance(lhs, NDArrayValue) and isinstance(rhs, NumberValue):
            return self.reduce_ndarray_and_number(reducer, lhs, rhs)
        elif isinstance(lhs, NDArrayValue) and isinstance(rhs, NDArrayValue):
            return self.reduce_ndarray_and_ndarray(reducer, lhs, rhs)
        raise TypeInferenceError(dbg, f"Operator `{self.get_name()}` not defined on `{lhs.type()}` and `{rhs.type()}`")
