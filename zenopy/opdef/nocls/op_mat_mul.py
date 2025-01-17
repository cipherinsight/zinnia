from typing import List, Dict, Optional

from zenopy.debug.exception import TypeInferenceError
from zenopy.opdef.nocls.abstract_op import AbstractOp
from zenopy.internal.dt_descriptor import DTDescriptor, FloatDTDescriptor, FloatType, IntegerType
from zenopy.debug.dbg_info import DebugInfo
from zenopy.builder.abstract_ir_builder import AbsIRBuilderInterface
from zenopy.builder.value import Value, NDArrayValue


class MatMulOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "mat_mul"

    @classmethod
    def get_name(cls) -> str:
        return "mat_mul"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("lhs"),
            AbstractOp._ParamEntry("rhs"),
        ]

    def _get_expected_result_dt(self, lhs_dt: DTDescriptor, rhs_dt: DTDescriptor):
        if isinstance(lhs_dt, FloatDTDescriptor) or isinstance(rhs_dt, FloatDTDescriptor):
            return FloatType
        return IntegerType

    def _reduce_constant_zero(self, reducer: AbsIRBuilderInterface, expected_dt: DTDescriptor):
        if expected_dt == FloatType:
            return reducer.ir_constant_float(0.0)
        return reducer.ir_constant_int(0)

    def build(self, reducer: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        lhs, rhs = kwargs["lhs"], kwargs["rhs"]
        if isinstance(lhs, NDArrayValue) and isinstance(rhs, NDArrayValue):
            if not NDArrayValue.matmul_compatible(lhs.shape(), rhs.shape()):
                raise TypeInferenceError(dbg, f'Invalid binary operator `{self.get_name()}` on operands {lhs.type()} and {rhs.type()}, as their shapes are not multiply compatible')
            expected_dtype = self._get_expected_result_dt(lhs.dtype(), rhs.dtype())
            return NDArrayValue.matmul(
                lhs, rhs,
                expected_dtype,
                lambda x, y: reducer.op_add(x, y),
                lambda x, y: reducer.op_multiply(x, y),
                lambda: self._reduce_constant_zero(reducer, expected_dtype)
            )
        raise TypeInferenceError(dbg, f'Invalid binary operator `{self.get_name()}` on operands {lhs.type()} and {rhs.type()}. Only ndarray can be passed to `{self.get_name()}`')
