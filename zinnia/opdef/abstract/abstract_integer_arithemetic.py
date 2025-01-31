from typing import Callable, Optional, Dict, List

from zinnia.debug.dbg_info import DebugInfo
from zinnia.debug.exception import TypeInferenceError
from zinnia.compile.type_sys import DTDescriptor, IntegerType
from zinnia.opdef.abstract.abstract_op import AbstractOp
from zinnia.compile.builder.abstract_ir_builder import AbsIRBuilderInterface
from zinnia.compile.builder.value import Value, IntegerValue, NDArrayValue


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

    def get_build_op_lambda(self, builder: AbsIRBuilderInterface) -> Callable[[IntegerValue, IntegerValue], IntegerValue]:
        raise NotImplementedError()

    def build_number_and_number(self, builder: AbsIRBuilderInterface, lhs: IntegerValue, rhs: IntegerValue) -> Value:
        return self.get_build_op_lambda(builder)(lhs, rhs)

    def build_number_and_ndarray(self, builder: AbsIRBuilderInterface, lhs: IntegerValue, rhs: NDArrayValue) -> Value:
        lhs_ndarray = NDArrayValue.from_number(lhs)
        lhs_ndarray, rhs_ndarray = NDArrayValue.binary_broadcast(lhs_ndarray, rhs)
        result = NDArrayValue.binary(lhs_ndarray, rhs_ndarray, IntegerType, self.get_build_op_lambda(builder))
        return result

    def build_ndarray_and_number(self, builder: AbsIRBuilderInterface, lhs: NDArrayValue, rhs: IntegerValue) -> Value:
        rhs_ndarray = NDArrayValue.from_number(rhs)
        lhs_ndarray, rhs_ndarray = NDArrayValue.binary_broadcast(lhs, rhs_ndarray)
        result = NDArrayValue.binary(lhs_ndarray, rhs_ndarray, IntegerType, self.get_build_op_lambda(builder))
        return result

    def build_ndarray_and_ndarray(self, builder: AbsIRBuilderInterface, lhs: NDArrayValue, rhs: NDArrayValue) -> NDArrayValue:
        if not NDArrayValue.binary_broadcast_compatible(lhs.shape(), rhs.shape()):
            raise TypeInferenceError(None, f"Cannot broadcast two NDArray with shapes {lhs.shape()} and {rhs.shape()}")
        lhs, rhs = NDArrayValue.binary_broadcast(lhs, rhs)
        result = NDArrayValue.binary(lhs, rhs, IntegerType, self.get_build_op_lambda(builder))
        return result

    def build(self, builder: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        lhs, rhs = kwargs["lhs"], kwargs["rhs"]
        if isinstance(lhs, IntegerValue) and isinstance(rhs, IntegerValue):
            return self.build_number_and_number(builder, lhs, rhs)
        elif isinstance(lhs, IntegerValue) and isinstance(rhs, NDArrayValue):
            return self.build_number_and_ndarray(builder, lhs, rhs)
        elif isinstance(lhs, NDArrayValue) and isinstance(rhs, IntegerValue):
            return self.build_ndarray_and_number(builder, lhs, rhs)
        elif isinstance(lhs, NDArrayValue) and isinstance(rhs, NDArrayValue):
            return self.build_ndarray_and_ndarray(builder, lhs, rhs)
        raise TypeInferenceError(dbg, f"Operator `{self.get_name()}` not defined on `{lhs.type()}` and `{rhs.type()}`")
