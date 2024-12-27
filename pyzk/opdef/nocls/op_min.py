from typing import Any, Optional, List, Dict

from pyzk.debug.dbg_info import DebugInfo
from pyzk.debug.exception import OperatorCallError, TypeInferenceError
from pyzk.internal.dt_descriptor import IntegerDTDescriptor, DTDescriptor, NumberDTDescriptor, \
    NDArrayDTDescriptor, TupleDTDescriptor, FloatDTDescriptor
from pyzk.internal.flatten_descriptor import FlattenDescriptor, IntegerFlattenDescriptor, FloatFlattenDescriptor
from pyzk.internal.inference_descriptor import InferenceDescriptor, NDArrayInferenceDescriptor, \
    FloatInferenceDescriptor, IntegerInferenceDescriptor, TupleInferenceDescriptor
from pyzk.opdef.nocls.abstract_op import AbstractOp


class MinOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "min"

    @classmethod
    def get_name(cls) -> str:
        return "min"

    def params_parse(self, dbg_i: Optional[DebugInfo], args: List[Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        if len(kwargs) > 0:
            raise OperatorCallError(dbg_i, f"Operator `{self.get_name()}` does not accept keyword arguments")
        if len(args) == 0:
            raise OperatorCallError(dbg_i, f"Operator `{self.get_name()}` requires at least one argument")
        return {f"_n_{i}": arg for i, arg in enumerate(args)}

    def type_check(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        args = [v for k, v in kwargs.items()]
        if len(args) == 1:
            n0 = args[0].dt
            if isinstance(n0, NumberDTDescriptor):
                raise TypeInferenceError(dbg_i, f"{n0} is not iterable")
            elif isinstance(n0, NDArrayDTDescriptor):
                if len(n0.shape) > 0:
                    raise TypeInferenceError(dbg_i, f"{n0} should be a 1-dim array here. Use NDArray.{self.get_name()} if you would like to perform axis-wise {self.get_name()}")
                return n0.dtype
            elif isinstance(n0, TupleDTDescriptor):
                return IntegerDTDescriptor()
            else:
                raise TypeInferenceError(dbg_i, f"Unsupported non-iterable datatype {n0}")
        else:
            has_ndarray = any([isinstance(arg.dt, NDArrayDTDescriptor) for arg in args])
            has_tuple = any([isinstance(arg.dt, TupleDTDescriptor) for arg in args])
            if has_ndarray:
                raise TypeInferenceError(dbg_i, f"Should not pass NDArray as argument here. To perform element-wise {self.get_name()}, use minimum instead")
            if has_tuple:
                raise TypeInferenceError(dbg_i, f"Tuple not allowed here for {self.get_name()}")
            same_type = all([arg.dt == args[0].dt for arg in args])
            if not same_type:
                raise TypeInferenceError(dbg_i, f"All number arguments for {self.get_name()} should have the same type")
            return args[0].dt

    def static_infer(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        args = [v for k, v in kwargs.items()]
        if len(args) == 1:
            n0 = args[0].dt
            if isinstance(n0, NDArrayInferenceDescriptor):
                assert len(n0.shape()) == 1
                val = n0.get().values[0]
                for v in n0.get().values[1:]:
                    if v is not None and val is not None and v < val:
                        val = v
                    elif v is None or val is None:
                        val = None
                if n0.dtype() == IntegerDTDescriptor():
                    return IntegerInferenceDescriptor(val)
                if n0.dtype() == FloatDTDescriptor():
                    return FloatInferenceDescriptor(val)
                raise NotImplementedError()
            elif isinstance(n0, TupleInferenceDescriptor):
                # Tuple elements are all integers
                val = n0.get()[0]
                for v in n0.get()[1:]:
                    if v is not None and val is not None and v < val:
                        val = v
                    elif v is None or val is None:
                        val = None
                return IntegerInferenceDescriptor(val)
        else:
            val = args[0].get()
            for v in args[1:]:
                if v is not None and val is not None and v.get() < val:
                    val = v.get()
                elif v is None or val is None:
                    val = None
            if args[0].dt == IntegerDTDescriptor():
                return IntegerInferenceDescriptor(val)
            if args[0].dt == FloatDTDescriptor():
                return FloatInferenceDescriptor(val)
        raise NotImplementedError()

    def ir_flatten(self, ir_builder, kwargs: Dict[str, FlattenDescriptor]) -> FlattenDescriptor:
        args = [v for k, v in kwargs.items()]
        if len(args) == 1:
            n0 = args[0].dt
            if isinstance(n0, NDArrayInferenceDescriptor):
                assert len(n0.shape()) == 1
                candidate = n0.get().values[0]
                if n0.dtype() == IntegerDTDescriptor():
                    for v in n0.get().values[1:]:
                        cond = ir_builder.create_less_than_i(candidate, v)
                        not_cond = ir_builder.create_logical_not(cond)
                        candidate = ir_builder.create_add_i(
                            ir_builder.create_mul_i(not_cond, v),
                            ir_builder.create_mul_i(cond, candidate)
                        )
                    return IntegerFlattenDescriptor(candidate)
                if n0.dtype() == FloatDTDescriptor():
                    for v in n0.get().values[1:]:
                        cond = ir_builder.create_less_than_f(candidate, v)
                        not_cond = ir_builder.create_logical_not(cond)
                        candidate = ir_builder.create_add_f(
                            ir_builder.create_mul_f(ir_builder.create_float_cast(not_cond), v),
                            ir_builder.create_mul_f(ir_builder.create_float_cast(cond), candidate)
                        )
                    return FloatFlattenDescriptor(candidate)
            raise NotImplementedError()
        else:
            candidate = args[0].ptr()
            if args[0].dt == IntegerDTDescriptor():
                for v in args[1:]:
                    cond = ir_builder.create_less_than_i(candidate, v.ptr())
                    not_cond = ir_builder.create_logical_not(cond)
                    candidate = ir_builder.create_add_i(
                        ir_builder.create_mul_i(not_cond, v.ptr()),
                        ir_builder.create_mul_i(cond, candidate)
                    )
                return IntegerFlattenDescriptor(candidate)
            if args[0].dt == FloatDTDescriptor():
                for v in args[1:]:
                    cond = ir_builder.create_less_than_f(candidate, v)
                    not_cond = ir_builder.create_logical_not(cond)
                    candidate = ir_builder.create_add_f(
                        ir_builder.create_mul_f(ir_builder.create_float_cast(not_cond), v),
                        ir_builder.create_mul_f(ir_builder.create_float_cast(cond), candidate)
                    )
                return FloatFlattenDescriptor(candidate)
            raise NotImplementedError()
