from typing import Callable, Optional, Dict, List

from zenopy.algo.ndarray_helper import NDArrayValueWrapper
from zenopy.debug.dbg_info import DebugInfo
from zenopy.debug.exception import TypeInferenceError
from zenopy.internal.dt_descriptor import DTDescriptor, IntegerType

from zenopy.opdef.nocls.abstract_op import AbstractOp
from zenopy.builder.abstract_ir_builder import AbsIRBuilderInterface
from zenopy.builder.value import Value, IntegerValue, NDArrayValue


class AbstractIntegerArithemetic(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("lhs"),
            AbstractOp._ParamEntry("rhs"),
        ]

    def check_ndarray_dtype(self, dbg_i: Optional[DebugInfo], dtype: DTDescriptor) -> None:
        if dtype != IntegerType:
            raise TypeInferenceError(dbg_i, f'The dtype of NDArray should be `Integer` in {self.get_name()}')

    def get_reduce_op_lambda(self, reducer: AbsIRBuilderInterface) -> Callable[[IntegerValue, IntegerValue], IntegerValue]:
        raise NotImplementedError()

    def reduce_number_and_number(self, reducer: AbsIRBuilderInterface, lhs: IntegerValue, rhs: IntegerValue) -> Value:
        return self.get_reduce_op_lambda(reducer)(lhs, rhs)

    def reduce_number_and_ndarray(self, reducer: AbsIRBuilderInterface, lhs: IntegerValue, rhs: NDArrayValue) -> Value:
        lhs_ndarray = NDArrayValue.from_number(lhs)
        lhs_ndarray, rhs_ndarray = NDArrayValue.broadcast(lhs_ndarray, rhs)
        result = NDArrayValue.binary(lhs_ndarray, rhs_ndarray, IntegerType, self.get_reduce_op_lambda(reducer))
        return result

    def reduce_ndarray_and_number(self, reducer: AbsIRBuilderInterface, lhs: NDArrayValue, rhs: IntegerValue) -> Value:
        rhs_ndarray = NDArrayValue.from_number(rhs)
        lhs_ndarray, rhs_ndarray = NDArrayValue.broadcast(lhs, rhs_ndarray)
        result = NDArrayValue.binary(lhs_ndarray, rhs_ndarray, IntegerType, self.get_reduce_op_lambda(reducer))
        return result

    def reduce_ndarray_and_ndarray(self, reducer: AbsIRBuilderInterface, lhs: NDArrayValue, rhs: NDArrayValue) -> NDArrayValue:
        if not NDArrayValueWrapper.binary_broadcast_compatible(lhs.shape(), rhs.shape()):
            raise TypeInferenceError(None, f"Cannot broadcast two NDArray with shapes {lhs.shape()} and {rhs.shape()}")
        lhs, rhs = NDArrayValue.broadcast(lhs, rhs)
        result = NDArrayValue.binary(lhs, rhs, IntegerType, self.get_reduce_op_lambda(reducer))
        return result

    def build(self, reducer: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        lhs, rhs = kwargs["lhs"], kwargs["rhs"]
        if isinstance(lhs, IntegerValue) and isinstance(rhs, IntegerValue):
            return self.reduce_number_and_number(reducer, lhs, rhs)
        elif isinstance(lhs, IntegerValue) and isinstance(rhs, NDArrayValue):
            return self.reduce_number_and_ndarray(reducer, lhs, rhs)
        elif isinstance(lhs, NDArrayValue) and isinstance(rhs, IntegerValue):
            return self.reduce_ndarray_and_number(reducer, lhs, rhs)
        elif isinstance(lhs, NDArrayValue) and isinstance(rhs, NDArrayValue):
            return self.reduce_ndarray_and_ndarray(reducer, lhs, rhs)
        raise TypeInferenceError(dbg, f"Operator `{self.get_name()}` not defined on `{lhs.type()}` and `{rhs.type()}`")
