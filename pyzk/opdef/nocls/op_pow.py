from typing import List, Dict, Optional, Tuple

from pyzk.algo.ndarray_helper import NDArrayHelper
from pyzk.debug.exception import TypeInferenceError, StaticInferenceError
from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.internal.dt_descriptor import DTDescriptor, IntegerDTDescriptor, FloatDTDescriptor, NDArrayDTDescriptor, \
    NumberDTDescriptor
from pyzk.internal.flatten_descriptor import FlattenDescriptor, IntegerFlattenDescriptor, FloatFlattenDescriptor, \
    NDArrayFlattenDescriptor
from pyzk.internal.inference_descriptor import InferenceDescriptor, IntegerInferenceDescriptor, \
    FloatInferenceDescriptor, NDArrayInferenceDescriptor, NDArrayInferenceValue, NumberInferenceDescriptor
from pyzk.debug.dbg_info import DebugInfo


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

    def _check_ndarray_no_negative(self, ndarray: NDArrayInferenceValue) -> bool:
        for val in ndarray.values:
            if val is not None and val < 0:
                return False
        return True

    def _get_result_shape(self, dbg_i: Optional[DebugInfo], x: Tuple[int, ...], exponent: Tuple[int, ...], mod: Tuple[int, ...]) -> Tuple[int, ...]:
        if not NDArrayHelper.broadcast_compatible(x, exponent):
            raise TypeInferenceError(dbg_i, f'Failed to broadcast on operand x and exponent')
        if not NDArrayHelper.broadcast_compatible(x, mod):
            raise TypeInferenceError(dbg_i, f'Failed to broadcast on operand x and mod')
        return NDArrayHelper.broadcast_shape(NDArrayHelper.broadcast_shape(x, exponent), mod)

    def _check_mod_allowed(self, x: DTDescriptor, exponent: DTDescriptor, mod: DTDescriptor | None) -> bool:
        if mod is not None:
            if not isinstance(x, IntegerDTDescriptor) or not isinstance(exponent, IntegerDTDescriptor):
                return False
            return isinstance(mod, IntegerDTDescriptor)
        return True

    def type_check(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        x = kwargs["x"]
        exponent = kwargs["exponent"]
        mod = kwargs["mod"]
        if not isinstance(x, NumberInferenceDescriptor) and not isinstance(x, NDArrayInferenceDescriptor):
            raise TypeInferenceError(dbg_i, f'Invalid type for operand `x`: {x.type()}')
        if not isinstance(exponent, NumberInferenceDescriptor) and not isinstance(exponent, NDArrayInferenceDescriptor):
            raise TypeInferenceError(dbg_i, f'Invalid type for operand `exponent`: {exponent.type()}')
        if mod is not None and not isinstance(mod, NumberInferenceDescriptor) and not isinstance(mod, NDArrayInferenceDescriptor):
            raise TypeInferenceError(dbg_i, f'Invalid type for operand `mod`: {mod.type()}')
        x_shape = x.shape() if isinstance(x, NDArrayInferenceDescriptor) else (1, )
        exponent_shape = exponent.shape() if isinstance(exponent, NDArrayInferenceDescriptor) else (1, )
        mod_shape = mod.shape() if mod is not None and isinstance(mod, NDArrayInferenceDescriptor) else (1, )
        x_dtype = x.dtype() if isinstance(x, NDArrayInferenceDescriptor) else x.dt
        exponent_dtype = exponent.dtype() if isinstance(exponent, NDArrayInferenceDescriptor) else exponent.dt
        mod_dtype = mod.dtype() if mod is not None and isinstance(mod, NDArrayInferenceDescriptor) else (mod.dt if mod is not None else mod)
        result_dtype = self._get_result_dtype(x_dtype, exponent_dtype)
        if not self._check_mod_allowed(x_dtype, exponent_dtype, mod_dtype):
            raise TypeInferenceError(dbg_i, f'pow() 3rd argument not allowed unless all arguments are integers')
        result_shape = self._get_result_shape(dbg_i, x_shape, exponent_shape, mod_shape)
        if result_shape != (1,) or isinstance(x.type(), NDArrayDTDescriptor):
            return NDArrayDTDescriptor(result_shape, result_dtype)
        return result_dtype

    def static_infer(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        x = kwargs["x"]
        exponent = kwargs["exponent"]
        mod = kwargs["mod"]
        x_shape = x.shape() if isinstance(x, NDArrayInferenceDescriptor) else (1, )
        exponent_shape = exponent.shape() if isinstance(exponent, NDArrayInferenceDescriptor) else (1, )
        mod_shape = mod.shape() if mod is not None and isinstance(mod, NDArrayInferenceDescriptor) else (1, )
        x_dtype = x.dtype() if isinstance(x, NDArrayInferenceDescriptor) else x.dt
        exponent_dtype = exponent.dtype() if isinstance(exponent, NDArrayInferenceDescriptor) else exponent.dt
        mod_dtype = mod.dtype() if mod is not None and isinstance(mod, NDArrayInferenceDescriptor) else (mod.dt if mod is not None else mod)
        result_dtype = self._get_result_dtype(x_dtype, exponent_dtype)
        result_shape = self._get_result_shape(dbg_i, x_shape, exponent_shape, mod_shape)
        if result_shape != (1,) or isinstance(x, NDArrayInferenceDescriptor):
            x_ndarray = x.get() if isinstance(x, NDArrayInferenceDescriptor) else NDArrayInferenceValue((1, ), [x.get()])
            exponent_ndarray = exponent.get() if isinstance(exponent, NDArrayInferenceDescriptor) else NDArrayInferenceValue((1, ), [exponent.get()])
            x_ndarray, exponent_ndarray = NDArrayHelper.broadcast(x_ndarray, exponent_ndarray)
            if mod is not None:
                mod_ndarray = mod.get() if isinstance(mod, NDArrayInferenceDescriptor) else NDArrayInferenceValue((1, ), [mod.get()])
                x_ndarray, mod_ndarray = NDArrayHelper.broadcast(x_ndarray, mod_ndarray)
                x_ndarray, exponent_ndarray = NDArrayHelper.broadcast(x_ndarray, exponent_ndarray)
                assert x_ndarray.shape == exponent_ndarray.shape == mod_ndarray.shape
                assert x_dtype == exponent_dtype == mod_dtype == IntegerDTDescriptor()
                if not self._check_ndarray_no_negative(exponent_ndarray):
                    raise StaticInferenceError(dbg_i, f'pow() 2nd argument must be non-negative if 3rd argument is present')
                return NDArrayInferenceDescriptor(
                    result_shape, result_dtype, x_ndarray.binary(
                        exponent_ndarray, lambda u, v: (u ** v) if u is not None and v is not None else None
                    ).binary(
                        mod_ndarray, lambda u, v: (u % v) if u is not None and v is not None else None
                    )
                )
            else:
                assert x_ndarray.shape == exponent_ndarray.shape
                if not self._check_ndarray_no_negative(exponent_ndarray):
                    raise StaticInferenceError(dbg_i, f'pow() 2nd argument must be non-negative if all arguments are Integer')
                if isinstance(result_dtype, IntegerDTDescriptor):
                    return NDArrayInferenceDescriptor(
                        result_shape, result_dtype, x_ndarray.binary(exponent_ndarray, lambda u, v: (u ** v) if u is not None and v is not None and v >= 0 else None)
                    )
                return NDArrayInferenceDescriptor(
                    result_shape, result_dtype, x_ndarray.binary(exponent_ndarray, lambda u, v: None)
                )
        x_val = x.get()
        exponent_val = exponent.get()
        if mod is not None:
            if exponent_val < 0:
                raise StaticInferenceError(dbg_i, f'pow() 2nd argument must be non-negative if 3rd argument is present')
            mod_val = mod.get()
            return IntegerInferenceDescriptor((x_val ** exponent_val) % mod_val)
        if isinstance(result_dtype, FloatDTDescriptor):
            return FloatInferenceDescriptor(None)
        elif isinstance(result_dtype, IntegerDTDescriptor):
            if exponent_val < 0:
                raise StaticInferenceError(dbg_i, f'pow() 2nd argument must be non-negative if all arguments are Integer')
            return IntegerInferenceDescriptor(x_val ** exponent_val)
        raise NotImplementedError()

    def ir_flatten(self, ir_builder, kwargs: Dict[str, FlattenDescriptor]) -> FlattenDescriptor:
        x = kwargs["x"]
        exponent = kwargs["exponent"]
        mod = kwargs["mod"]
        x_shape = x.shape() if isinstance(x, NDArrayFlattenDescriptor) else (1, )
        exponent_shape = exponent.shape() if isinstance(exponent, NDArrayFlattenDescriptor) else (1, )
        mod_shape = mod.shape() if mod is not None and isinstance(mod, NDArrayFlattenDescriptor) else (1, )
        x_dtype = x.dtype() if isinstance(x, NDArrayFlattenDescriptor) else x.dt
        exponent_dtype = exponent.dtype() if isinstance(exponent, NDArrayFlattenDescriptor) else exponent.dt
        mod_dtype = mod.dtype() if isinstance(mod, NDArrayFlattenDescriptor) else (mod.dt if mod is not None else mod)
        result_dtype = self._get_result_dtype(x_dtype, exponent_dtype)
        result_shape = self._get_result_shape(None, x_shape, exponent_shape, mod_shape)
        if result_shape != (1,) or isinstance(x, NDArrayFlattenDescriptor):
            x_ndarray = x.ptr() if isinstance(x, NDArrayFlattenDescriptor) else NDArrayInferenceValue((1, ), [x.ptr()])
            exponent_ndarray = exponent.ptr() if isinstance(exponent, NDArrayFlattenDescriptor) else NDArrayInferenceValue((1, ), [exponent.ptr()])
            x_ndarray, exponent_ndarray = NDArrayHelper.broadcast(x_ndarray, exponent_ndarray)
            if mod is not None:
                mod_ndarray = mod.ptr() if isinstance(mod, NDArrayFlattenDescriptor) else NDArrayInferenceValue((1, ), [mod.ptr()])
                x_ndarray, mod_ndarray = NDArrayHelper.broadcast(x_ndarray, mod_ndarray)
                x_ndarray, exponent_ndarray = NDArrayHelper.broadcast(x_ndarray, exponent_ndarray)
                assert x_ndarray.shape == exponent_ndarray.shape == mod_ndarray.shape
                assert x_dtype == exponent_dtype == mod_dtype == IntegerDTDescriptor()
                exponent_ndarray.unary(lambda u: ir_builder.create_assert(ir_builder.create_greater_than_or_equal_i(u, ir_builder.create_constant(0))))
                return NDArrayFlattenDescriptor(
                    result_shape, result_dtype, x_ndarray.binary(
                        exponent_ndarray, lambda u, v: ir_builder.create_pow_i(u, v)
                    ).binary(
                        mod_ndarray, lambda u, v: ir_builder.create_mod_i(u, v)
                    )
                )
            else:
                if isinstance(exponent, IntegerDTDescriptor):
                    exponent_ndarray.unary(lambda u: ir_builder.create_assert(
                        ir_builder.create_greater_than_or_equal_i(u, ir_builder.create_constant(0))))
                elif isinstance(exponent, FloatDTDescriptor):
                    exponent_ndarray.unary(lambda u: ir_builder.create_assert(
                        ir_builder.create_greater_than_or_equal_f(u, ir_builder.create_constant_float(0))))
                assert x_ndarray.shape == exponent_ndarray.shape
                if isinstance(result_dtype, IntegerDTDescriptor):
                    return NDArrayFlattenDescriptor(
                        result_shape, result_dtype, x_ndarray.binary(
                            exponent_ndarray, lambda u, v: ir_builder.create_pow_i(u, v)
                        )
                    )
                return NDArrayFlattenDescriptor(
                    result_shape, result_dtype, x_ndarray.binary(exponent_ndarray, lambda u, v: ir_builder.create_pow_f(
                        u if x_dtype == FloatDTDescriptor() else ir_builder.create_float_cast(u), v if exponent_dtype == FloatDTDescriptor() else ir_builder.create_float_cast(v)
                    ))
                )
        x_val = x.ptr()
        exponent_val = exponent.ptr()
        if mod is not None:
            ir_builder.create_assert(ir_builder.create_greater_than_or_equal_i(exponent_val, ir_builder.create_constant(0)))
            mod_val = mod.ptr()
            return IntegerFlattenDescriptor(ir_builder.create_mod_i(ir_builder.create_pow_i(x_val, exponent_val), mod_val))
        if isinstance(result_dtype, FloatDTDescriptor):
            return FloatFlattenDescriptor(ir_builder.create_pow_f(
                x_val if x_dtype == FloatDTDescriptor() else ir_builder.create_float_cast(x_val), exponent_val if exponent_dtype == FloatDTDescriptor() else ir_builder.create_float_cast(exponent_val)
            ))
        elif isinstance(result_dtype, IntegerDTDescriptor):
            ir_builder.create_assert(ir_builder.create_greater_than_or_equal_i(exponent_val, ir_builder.create_constant(0)))
            return IntegerFlattenDescriptor(ir_builder.create_pow_i(x_val, exponent_val))
        raise NotImplementedError()
