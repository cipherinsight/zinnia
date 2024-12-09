from typing import Dict, List, Any, Optional

from pyzk.debug.exception import TypeInferenceError, StaticInferenceError
from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.internal.dt_descriptor import DTDescriptor, IntegerDTDescriptor, NDArrayDTDescriptor, FloatDTDescriptor
from pyzk.internal.flatten_descriptor import FlattenDescriptor, NDArrayFlattenDescriptor, IntegerFlattenDescriptor, \
    FloatFlattenDescriptor
from pyzk.internal.inference_descriptor import InferenceDescriptor, NDArrayInferenceDescriptor, \
    IntegerInferenceDescriptor, FloatInferenceDescriptor
from pyzk.algo.ndarray_helper import NDArrayHelper
from pyzk.debug.dbg_info import DebugInfo


class AbstractAggregator(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("self"),
            AbstractOp._ParamEntry("axis", default=True),
        ]

    def aggregator_func(self, lhs: Any, rhs: Any, dt: DTDescriptor) -> Any:
        raise NotImplementedError()

    def initial_func(self, dt: DTDescriptor) -> Any:
        raise NotImplementedError()

    def aggregator_build_ir(self, ir_builder, lhs: int, rhs: int, dt: DTDescriptor) -> int:
        raise NotImplementedError()

    def initial_build_ir(self, ir_builder, dt: DTDescriptor) -> int:
        raise NotImplementedError()

    def get_result_dtype(self, element_dt: DTDescriptor):
        return element_dt

    def is_allowed_ndarray_dtype(self, element_dt: DTDescriptor) -> bool:
        return True

    def type_check(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        the_self = kwargs["self"]
        the_axis = kwargs["axis"]
        dtype = the_self.dtype()
        if not isinstance(the_self, NDArrayInferenceDescriptor):
            raise TypeInferenceError(dbg_i, "Param `self` must be of type `NDArray`")
        if not self.is_allowed_ndarray_dtype(dtype):
            raise TypeInferenceError(dbg_i, f"The dtype ({dtype}) of param `self: NDArray` is not allowed here")
        if the_axis is None:
            _axis = -1
        elif not isinstance(the_axis.type(), IntegerDTDescriptor):
            raise TypeInferenceError(dbg_i, "Param `axis` must be of type `Integer`")
        elif the_axis.get() is None:
            raise StaticInferenceError(dbg_i, "Cannot statically infer the value of param `axis`")
        else:
            _axis = the_axis.get()
        if _axis >= len(the_self.shape()):
            raise TypeInferenceError(dbg_i, f"Invalid `axis` value for `{self.get_signature()}`. The axis number exceeds total number of dimensions of the ndarray")
        if _axis == -1:
            return self.get_result_dtype(dtype)
        the_shape = the_self.shape()
        new_shape = tuple(x for i, x in enumerate(the_shape) if i != _axis)
        return NDArrayDTDescriptor(new_shape, self.get_result_dtype(dtype))

    def static_infer(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        the_self = kwargs["self"]
        the_axis = kwargs["axis"]
        dtype = the_self.dtype()
        inferred_value = the_self.get().accumulate(
            the_axis.get() if the_axis is not None else -1,
            lambda x, y: self.aggregator_func(x, y, dtype),
            lambda: self.initial_func(dtype)
        )
        if not isinstance(inferred_value, NDArrayHelper):
            if isinstance(self.get_result_dtype(dtype), IntegerDTDescriptor):
                return IntegerInferenceDescriptor(inferred_value)
            elif isinstance(self.get_result_dtype(dtype), FloatDTDescriptor):
                return FloatInferenceDescriptor(inferred_value)
            raise NotImplementedError()
        return NDArrayInferenceDescriptor(inferred_value.shape, self.get_result_dtype(dtype), inferred_value)

    def ir_flatten(self, ir_builder, kwargs: Dict[str, FlattenDescriptor]) -> FlattenDescriptor:
        the_self = kwargs["self"]
        the_axis = kwargs["axis"]
        dtype = the_self.dtype()
        inferred_value = the_self.ptr().accumulate(
            the_axis.val() if the_axis is not None else -1,
            lambda x, y: self.aggregator_build_ir(ir_builder, x, y, dtype),
            lambda: self.initial_build_ir(ir_builder, dtype)
        )
        if not isinstance(inferred_value, NDArrayHelper):
            if isinstance(self.get_result_dtype(dtype), IntegerDTDescriptor):
                return IntegerFlattenDescriptor(inferred_value)
            elif isinstance(self.get_result_dtype(dtype), FloatDTDescriptor):
                return FloatFlattenDescriptor(inferred_value)
            raise NotImplementedError()
        return NDArrayFlattenDescriptor(inferred_value.shape, self.get_result_dtype(dtype), inferred_value)

