from typing import Dict, List, Optional, Tuple

from pyzk.debug.exception import TypeInferenceError, StaticInferenceError
from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.internal.dt_descriptor import DTDescriptor
from pyzk.algo.ndarray_helper import NDArrayValueWrapper
from pyzk.debug.dbg_info import DebugInfo
from pyzk.builder.abstract_ir_builder import AbsIRBuilderInterface
from pyzk.builder.value import Value, NDArrayValue, IntegerValue, NumberValue


class AbstractAggregator(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("self"),
            AbstractOp._ParamEntry("axis", default=True),
        ]

    def aggregator_func(self, reducer: AbsIRBuilderInterface, lhs: NumberValue, lhs_i: NumberValue, rhs: NumberValue, rhs_i: NumberValue, dt: DTDescriptor) -> Tuple[NumberValue, NumberValue | None]:
        raise NotImplementedError()

    def initial_func(self, reducer: AbsIRBuilderInterface, dt: DTDescriptor, first_ele: NumberValue) -> Tuple[NumberValue, NumberValue | None]:
        raise NotImplementedError()

    def depair_func(self, reducer: AbsIRBuilderInterface, a: NumberValue, b: NumberValue) -> NumberValue:
        return a

    def enpair_func(self, reducer: AbsIRBuilderInterface, a: NumberValue, b: int) -> Tuple[NumberValue, NumberValue | None]:
        return a, None

    def get_result_dtype(self, element_dt: DTDescriptor):
        return element_dt

    def is_allowed_ndarray_dtype(self, element_dt: DTDescriptor) -> bool:
        return True

    def build(self, reducer: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        the_self = kwargs["self"]
        the_axis = kwargs["axis"]
        assert isinstance(the_self, NDArrayValue)
        dtype = the_self.dtype()
        if not self.is_allowed_ndarray_dtype(dtype):
            raise TypeInferenceError(dbg, f"The dtype ({dtype}) of param `self: NDArray` is not allowed here")
        if the_axis is None:
            _axis = -1
        elif not isinstance(the_axis, IntegerValue):
            raise TypeInferenceError(dbg, "Param `axis` must be of type `Integer`")
        elif the_axis.val() is None:
            raise StaticInferenceError(dbg, "Cannot statically infer the value of param `axis`")
        else:
            _axis = the_axis.val()
        if _axis >= len(the_self.shape()):
            raise TypeInferenceError(dbg, f"Invalid `axis` value for `{self.get_signature()}`. The axis number exceeds total number of dimensions of the ndarray")
        dtype = self.get_result_dtype(dtype)
        result_value = the_self.get().accumulate(
            _axis,
            lambda x, x_i, y, y_i: self.aggregator_func(reducer, x, x_i, y, y_i, dtype),
            lambda first_ele: self.initial_func(reducer, dtype, first_ele),
            lambda x, y: self.enpair_func(reducer, x, y),
            lambda x, y: self.depair_func(reducer, x, y)
        )
        if isinstance(result_value, NDArrayValueWrapper):
            return NDArrayValue(result_value.shape, dtype, result_value)
        else:
            return result_value
