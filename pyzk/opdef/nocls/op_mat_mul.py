from typing import List, Dict, Optional

from pyzk.debug.exception import TypeInferenceError
from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.internal.dt_descriptor import DTDescriptor, NDArrayDTDescriptor, FloatDTDescriptor, IntegerDTDescriptor
from pyzk.internal.flatten_descriptor import FlattenDescriptor, NDArrayFlattenDescriptor
from pyzk.internal.inference_descriptor import NDArrayInferenceDescriptor, InferenceDescriptor
from pyzk.algo.ndarray_helper import NDArrayHelper
from pyzk.debug.dbg_info import DebugInfo


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
        if isinstance(lhs, NDArrayDTDescriptor) and isinstance(rhs, NDArrayDTDescriptor):
            if not NDArrayHelper.matmul_shape_matches(lhs.shape, rhs.shape):
                raise TypeInferenceError(dbg_i, f'Invalid binary operator `{self.get_signature()}` on operands {lhs} and {rhs}, as their shapes are not multiply compatible')
            return NDArrayDTDescriptor(shape=NDArrayHelper.matmul_shape(lhs.shape, rhs.shape), dtype=self.get_expected_result_dt(lhs.dtype, rhs.dtype))
        raise TypeInferenceError(dbg_i, f'Invalid binary operator `{self.get_signature()}` on operands {lhs} and {rhs}. Only ndarray can be passed to `{self.get_signature()}`')

    def static_infer(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        lhs, rhs = kwargs["lhs"], kwargs["rhs"]
        if isinstance(lhs, NDArrayInferenceDescriptor) and isinstance(rhs, NDArrayInferenceDescriptor):
            matmul_shape = NDArrayHelper.matmul_shape(lhs.shape(), rhs.shape())
            return NDArrayInferenceDescriptor(
                matmul_shape,
                self.get_expected_result_dt(lhs.dtype(), rhs.dtype()),
                NDArrayHelper.matmul(
                    lhs.get(), rhs.get(),
                    lambda x, y: self._infer_add(x, y, lhs.dtype(), rhs.dtype()),
                    lambda x, y: self._infer_mul(x, y, lhs.dtype(), rhs.dtype()),
                    lambda: self._infer_constant_zero(lhs.dtype(), rhs.dtype())
                )
            )
        raise NotImplementedError()

    def ir_flatten(self, ir_builder, kwargs: Dict[str, FlattenDescriptor]) -> FlattenDescriptor:
        lhs, rhs = kwargs["lhs"], kwargs["rhs"]
        if isinstance(lhs, NDArrayFlattenDescriptor) and isinstance(rhs, NDArrayFlattenDescriptor):
            matmul_shape = NDArrayHelper.matmul_shape(lhs.shape(), rhs.shape())
            return NDArrayFlattenDescriptor(
                matmul_shape,
                self.get_expected_result_dt(lhs.dtype(), rhs.dtype()),
                NDArrayHelper.matmul(
                    lhs.ptr(), rhs.ptr(),
                    lambda x, y: self._flatten_add(ir_builder, x, y, lhs.dtype(), rhs.dtype()),
                    lambda x, y: self._flatten_mul(ir_builder, x, y, lhs.dtype(), rhs.dtype()),
                    lambda: self._flatten_constant_zero(ir_builder, lhs.dtype(), rhs.dtype())
                )
            )
        raise NotImplementedError()

    def _infer_add(self, x: int | float, y: int | float, x_dt: DTDescriptor, y_dt: DTDescriptor):
        if isinstance(x_dt, FloatDTDescriptor) or isinstance(y_dt, FloatDTDescriptor):
            return None
        if x is None or y is None:
            return None
        return x + y

    def _infer_mul(self, x: int | float, y: int | float, x_dt: DTDescriptor, y_dt: DTDescriptor):
        if isinstance(x_dt, FloatDTDescriptor) or isinstance(y_dt, FloatDTDescriptor):
            return None
        if x is None or y is None:
            return None
        return x * y

    def _infer_constant_zero(self, x_dt: DTDescriptor, y_dt: DTDescriptor):
        if isinstance(x_dt, FloatDTDescriptor) or isinstance(y_dt, FloatDTDescriptor):
            return 0.0
        return 0

    def _flatten_add(self, ir_builder, x: int, y: int, x_dt: DTDescriptor, y_dt: DTDescriptor):
        if isinstance(x_dt, FloatDTDescriptor) and isinstance(y_dt, FloatDTDescriptor):
            return ir_builder.create_add_f(x, y)
        elif isinstance(x_dt, IntegerDTDescriptor) and isinstance(y_dt, FloatDTDescriptor):
            return ir_builder.create_add_f(ir_builder.create_float_cast(x), y)
        elif isinstance(x_dt, FloatDTDescriptor) and isinstance(y_dt, IntegerDTDescriptor):
            return ir_builder.create_add_f(x, ir_builder.create_float_cast(y))
        elif isinstance(x_dt, IntegerDTDescriptor) and isinstance(y_dt, IntegerDTDescriptor):
            return ir_builder.create_add_i(x, y)
        raise NotImplementedError()

    def _flatten_mul(self, ir_builder, x: int, y: int, x_dt: DTDescriptor, y_dt: DTDescriptor):
        if isinstance(x_dt, FloatDTDescriptor) and isinstance(y_dt, FloatDTDescriptor):
            return ir_builder.create_mul_f(x, y)
        elif isinstance(x_dt, IntegerDTDescriptor) and isinstance(y_dt, FloatDTDescriptor):
            return ir_builder.create_mul_f(ir_builder.create_float_cast(x), y)
        elif isinstance(x_dt, FloatDTDescriptor) and isinstance(y_dt, IntegerDTDescriptor):
            return ir_builder.create_mul_f(x, ir_builder.create_float_cast(y))
        elif isinstance(x_dt, IntegerDTDescriptor) and isinstance(y_dt, IntegerDTDescriptor):
            return ir_builder.create_mul_i(x, y)
        raise NotImplementedError()

    def _flatten_constant_zero(self, ir_builder, x_dt: DTDescriptor, y_dt: DTDescriptor):
        if isinstance(x_dt, FloatDTDescriptor) or isinstance(y_dt, FloatDTDescriptor):
            return ir_builder.create_float_cast(ir_builder.create_constant(0))
        return ir_builder.create_constant(0)
