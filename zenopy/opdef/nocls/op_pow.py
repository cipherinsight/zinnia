from typing import List, Dict, Optional

from zenopy.algo.ndarray_helper import NDArrayValueWrapper
from zenopy.debug.exception import TypeInferenceError, StaticInferenceError
from zenopy.opdef.nocls.abstract_op import AbstractOp
from zenopy.internal.dt_descriptor import DTDescriptor, IntegerDTDescriptor, FloatDTDescriptor, IntegerType, FloatType, \
    NumberDTDescriptor
from zenopy.debug.dbg_info import DebugInfo
from zenopy.builder.abstract_ir_builder import AbsIRBuilderInterface
from zenopy.builder.value import Value, NumberValue, NDArrayValue, IntegerValue, FloatValue


class PowOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "pow"

    @classmethod
    def get_name(cls) -> str:
        return "pow"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("x"),
            AbstractOp._ParamEntry("exponent"),
            AbstractOp._ParamEntry("mod", default=True)
        ]

    def _get_result_dtype(self, x: DTDescriptor, exponent: DTDescriptor) -> DTDescriptor:
        if isinstance(x, IntegerDTDescriptor):
            if isinstance(exponent, IntegerDTDescriptor):
                return IntegerDTDescriptor()
            elif isinstance(exponent, FloatDTDescriptor):
                return FloatDTDescriptor()
        elif isinstance(x, FloatDTDescriptor):
            return FloatDTDescriptor()
        raise NotImplementedError()

    def _check_ndarray_no_negative(self, ndarray: NDArrayValue) -> bool:
        for val in ndarray.flattened_values():
            if val.val() is not None and val.val() < 0:
                return False
        return True

    def _check_mod_allowed(self, x: DTDescriptor, exponent: DTDescriptor, mod: DTDescriptor | None) -> bool:
        if mod is not None:
            if x != IntegerType or exponent != IntegerType:
                return False
            return mod == IntegerType
        return True

    def _reduce_pow(self, reducer: AbsIRBuilderInterface, x: NumberValue, exponent: NumberValue) -> NumberValue:
        if isinstance(x, IntegerValue) and isinstance(exponent, IntegerValue):
            return reducer.ir_pow_i(x, exponent)
        elif isinstance(x, FloatValue) or isinstance(exponent, FloatValue):
            return reducer.ir_pow_f(x, exponent)
        elif isinstance(x, IntegerValue) and isinstance(exponent, FloatValue):
            return reducer.ir_pow_f(reducer.ir_float_cast(x), exponent)
        elif isinstance(x, FloatValue) and isinstance(exponent, IntegerValue):
            return reducer.ir_pow_f(x, reducer.ir_float_cast(exponent))
        raise NotImplementedError()

    def _get_expected_result_dtype(self, x: NumberDTDescriptor, y: NumberDTDescriptor) -> NumberDTDescriptor:
        if x == FloatType or y == FloatType:
            return FloatType
        return IntegerType

    def build(self, reducer: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        x = kwargs["x"]
        exponent = kwargs["exponent"]
        mod = kwargs["mod"]
        if not isinstance(x, NumberValue) and not isinstance(x, NDArrayValue):
            raise TypeInferenceError(dbg, f'Invalid type for operand `x`: {x.type()}')
        if not isinstance(exponent, NumberValue) and not isinstance(exponent, NDArrayValue):
            raise TypeInferenceError(dbg, f'Invalid type for operand `exponent`: {exponent.type()}')
        if mod is not None and not isinstance(mod, NumberValue) and not isinstance(mod, NDArrayValue):
            raise TypeInferenceError(dbg, f'Invalid type for operand `mod`: {mod.type()}')
        x_shape = x.shape() if isinstance(x, NDArrayValue) else (1, )
        exponent_shape = exponent.shape() if isinstance(exponent, NDArrayValue) else (1, )
        mod_shape = mod.shape() if mod is not None and isinstance(mod, NDArrayValue) else (1, )
        x_dtype = x.dtype() if isinstance(x, NDArrayValue) else x.type()
        exponent_dtype = exponent.dtype() if isinstance(exponent, NDArrayValue) else exponent.type()
        mod_dtype = mod.dtype() if (mod is not None and isinstance(mod, NDArrayValue)) else (mod.type() if mod is not None else mod)
        result_dtype = self._get_result_dtype(x_dtype, exponent_dtype)
        if not self._check_mod_allowed(x_dtype, exponent_dtype, mod_dtype):
            raise TypeInferenceError(dbg, f'pow() 3rd argument not allowed unless all arguments are integers')
        if not NDArrayValueWrapper.binary_broadcast_compatible(x.shape(), exponent.shape()):
            raise TypeInferenceError(dbg, f'Failed to broadcast on operand x and exponent')
        if not NDArrayValueWrapper.binary_broadcast_compatible(NDArrayValueWrapper.binary_broadcast_shape(x_shape, exponent_shape), mod_shape):
            raise TypeInferenceError(dbg, f'Failed to broadcast on operand x, exponent and mod')
        result_is_ndarray = isinstance(x, NDArrayValue) or isinstance(exponent, NDArrayValue) or (mod is not None and isinstance(mod, NDArrayValue))
        x_ndarray = x.get() if isinstance(x, NDArrayValue) else NDArrayValue.from_number(x)
        exponent_ndarray = exponent.get() if isinstance(exponent, NDArrayValue) else NDArrayValue.from_number(exponent)
        mod_ndarray = mod.get() if (mod is not None and isinstance(mod, NDArrayValue)) else NDArrayValue.from_number(mod) if mod is not None else None
        if mod is not None and not self._check_ndarray_no_negative(exponent_ndarray):
            raise StaticInferenceError(dbg, f'pow() 2nd argument must be non-negative if 3rd argument is present')
        if result_dtype == IntegerType and not self._check_ndarray_no_negative(exponent_ndarray):
            raise StaticInferenceError(dbg, f'pow() 2nd argument must be non-negative if all arguments are Integer')
        x_ndarray, exponent_ndarray = NDArrayValue.broadcast(x_ndarray, exponent_ndarray)
        result = NDArrayValue.binary(x_ndarray, exponent_ndarray, self._get_expected_result_dtype(x_dtype, exponent_dtype), lambda u, v: self._reduce_pow(reducer, u, v))
        if mod is not None:
            assert result.dtype() == IntegerType
            result, mod_ndarray = NDArrayValue.broadcast(result, mod_ndarray)
            result = NDArrayValue.binary(result, mod_ndarray, IntegerType, lambda u, v: reducer.op_modulo(u, v))
        if not result_is_ndarray:
            result = result.flattened_values()
            assert len(result) == 1
            result = result[0]
        return result
