from typing import List, Dict, Callable, Optional

from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.debug.exception import TypeInferenceError
from zinnia.op_def.abstract.abstract_op import AbstractOp
from zinnia.compile.type_sys import DTDescriptor, IntegerDTDescriptor, FloatDTDescriptor, FloatType, IntegerType
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import NDArrayValue, NumberValue, Value


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

    def get_build_op_lambda(
            self,
            builder: IRBuilderInterface,
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

    def build_number_and_number(self, builder: IRBuilderInterface, lhs: NumberValue, rhs: NumberValue, dbg: Optional[DebugInfo] = None) -> Value:
        return self.get_build_op_lambda(builder)(lhs, rhs)

    def build_number_and_ndarray(self, builder: IRBuilderInterface, lhs: NumberValue, rhs: NDArrayValue, dbg: Optional[DebugInfo] = None) -> Value:
        lhs_ndarray = NDArrayValue.from_number(lhs)
        lhs_ndarray, rhs_ndarray = NDArrayValue.binary_broadcast(lhs_ndarray, rhs)
        result = NDArrayValue.binary(lhs_ndarray, rhs_ndarray, self.get_expected_result_dt(lhs.type(), rhs.dtype()), self.get_build_op_lambda(builder))
        return result

    def build_ndarray_and_number(self, builder: IRBuilderInterface, lhs: NDArrayValue, rhs: NumberValue, dbg: Optional[DebugInfo] = None) -> Value:
        rhs_ndarray = NDArrayValue.from_number(rhs)
        lhs_ndarray, rhs_ndarray = NDArrayValue.binary_broadcast(lhs, rhs_ndarray)
        result = NDArrayValue.binary(lhs_ndarray, rhs_ndarray, self.get_expected_result_dt(lhs.dtype(), rhs.type()), self.get_build_op_lambda(builder))
        return result

    def build_ndarray_and_ndarray(self, builder: IRBuilderInterface, lhs: NDArrayValue, rhs: NDArrayValue, dbg: Optional[DebugInfo] = None) -> NDArrayValue:
        if not NDArrayValue.binary_broadcast_compatible(lhs.shape(), rhs.shape()):
            raise TypeInferenceError(dbg, f"Cannot broadcast two NDArray with shapes {lhs.shape()} and {rhs.shape()}")
        lhs, rhs = NDArrayValue.binary_broadcast(lhs, rhs)
        result = NDArrayValue.binary(lhs, rhs, self.get_expected_result_dt(lhs.dtype(), rhs.dtype()), self.get_build_op_lambda(builder))
        return result

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        lhs, rhs = kwargs["lhs"], kwargs["rhs"]
        if isinstance(lhs, NumberValue) and isinstance(rhs, NumberValue):
            return self.build_number_and_number(builder, lhs, rhs, dbg)
        elif isinstance(lhs, NumberValue) and isinstance(rhs, NDArrayValue):
            return self.build_number_and_ndarray(builder, lhs, rhs, dbg)
        elif isinstance(lhs, NDArrayValue) and isinstance(rhs, NumberValue):
            return self.build_ndarray_and_number(builder, lhs, rhs, dbg)
        elif isinstance(lhs, NDArrayValue) and isinstance(rhs, NDArrayValue):
            return self.build_ndarray_and_ndarray(builder, lhs, rhs, dbg)
        raise TypeInferenceError(dbg, f"Operator `{self.get_name()}` not defined on `{lhs.type()}` and `{rhs.type()}`")
