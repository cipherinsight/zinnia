from typing import List, Optional, Tuple, cast

from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.compile.triplet import DynamicNDArrayValue, IntegerValue, NDArrayValue, NoneValue, NumberValue, Value
from zinnia.compile.type_sys.dt_descriptor import DTDescriptor
from zinnia.compile.type_sys.ndarray_bounds import infer_ndarray_compile_bounds_from_static_shape
from zinnia.debug.dbg_info import DebugInfo
from zinnia.debug.exception import StaticInferenceError, TypeInferenceError
from zinnia.op_def.abstract.abstract_op import AbstractOp
from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface


class AbstractAggregator(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("self"),
            AbstractOp._ParamEntry("axis", default=True),
        ]

    def aggregator_func(
        self,
        builder: IRBuilderInterface,
        lhs: NumberValue,
        lhs_i: NumberValue,
        rhs: NumberValue,
        rhs_i: NumberValue,
        dt: DTDescriptor,
    ) -> Tuple[NumberValue, NumberValue | None]:
        raise NotImplementedError()

    def initial_func(self, builder: IRBuilderInterface, dt: DTDescriptor, first_ele: NumberValue) -> Tuple[NumberValue, NumberValue | None]:
        raise NotImplementedError()

    def depair_func(self, builder: IRBuilderInterface, a: NumberValue, b: NumberValue) -> NumberValue:
        return a

    def enpair_func(self, builder: IRBuilderInterface, a: NumberValue, b: int) -> Tuple[NumberValue, NumberValue | None]:
        return a, None

    def get_result_dtype(self, element_dt: DTDescriptor):
        return element_dt

    def is_allowed_ndarray_dtype(self, element_dt: DTDescriptor) -> bool:
        return True

    def _offload_to_dynamic_aggregator(
        self,
        builder: IRBuilderInterface,
        the_self: NDArrayValue,
        the_axis: Value,
        dbg: Optional[DebugInfo],
    ) -> Value:
        from zinnia.op_def.dynamic_ndarray import (
            DynamicNDArray_SumOp,
            DynamicNDArray_ProdOp,
            DynamicNDArray_MaxOp,
            DynamicNDArray_MinOp,
            DynamicNDArray_ArgMaxOp,
            DynamicNDArray_ArgMinOp,
            DynamicNDArray_AllOp,
            DynamicNDArray_AnyOp,
        )

        op_lookup = {
            "sum": DynamicNDArray_SumOp,
            "prod": DynamicNDArray_ProdOp,
            "max": DynamicNDArray_MaxOp,
            "min": DynamicNDArray_MinOp,
            "argmax": DynamicNDArray_ArgMaxOp,
            "argmin": DynamicNDArray_ArgMinOp,
            "all": DynamicNDArray_AllOp,
            "any": DynamicNDArray_AnyOp,
        }
        op_cls = op_lookup.get(self.get_name())
        if op_cls is None:
            raise TypeInferenceError(dbg, f"No dynamic aggregator found for `{self.get_name()}`")

        dyn_self = the_self if isinstance(the_self, DynamicNDArrayValue) else the_self.to_dynamic_ndarray()
        op = op_cls()
        kwargs = op.argparse(dbg, [dyn_self], {"axis": the_axis})
        return op.build(builder, OpArgsContainer(kwargs), dbg)

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        the_self = kwargs["self"]
        the_axis = kwargs.get("axis", builder.op_constant_none())
        if the_axis is None:
            the_axis = builder.op_constant_none()
        assert isinstance(the_self, NDArrayValue)

        dtype = the_self.dtype()
        if not self.is_allowed_ndarray_dtype(dtype):
            raise TypeInferenceError(dbg, f"The dtype ({dtype}) of param `self: NDArray` is not allowed here")

        # All dynamic-ndarray reduction logic lives in dynamic_ndarray package.
        if isinstance(the_self, DynamicNDArrayValue):
            return self._offload_to_dynamic_aggregator(builder, the_self, the_axis, dbg)

        if isinstance(the_axis, IntegerValue) and the_axis.val(builder) is None:
            return self._offload_to_dynamic_aggregator(builder, the_self, the_axis, dbg)

        bounds = infer_ndarray_compile_bounds_from_static_shape(the_self.shape(), dbg, self.get_name())
        if isinstance(the_axis, NoneValue):
            _axis = None
        elif not isinstance(the_axis, IntegerValue):
            raise TypeInferenceError(dbg, "Param `axis` must be of type `Integer`")
        elif the_axis.val(builder) is None:
            raise StaticInferenceError(dbg, "Cannot statically infer the value of param `axis`")
        else:
            _axis = the_axis.val(builder)

        if _axis is not None:
            if _axis < 0:
                _axis += len(the_self.shape())
            if _axis < 0 or _axis >= len(the_self.shape()):
                raise TypeInferenceError(dbg, f"axis `{_axis}` is out of bounds for array of dimension {len(the_self.shape())}")

        max_reduction_steps = bounds.max_length if _axis is None else bounds.static_shape[_axis]
        if max_reduction_steps <= 0:
            raise TypeInferenceError(dbg, "Axis-wise reduction requires a positive static computation bound")

        dtype = self.get_result_dtype(dtype)
        return the_self.accumulate(
            cast(int, _axis),
            lambda x, x_i, y, y_i: self.aggregator_func(builder, x, x_i, y, y_i, dtype),
            lambda first_ele: self.initial_func(builder, dtype, first_ele),
            lambda x, y: self.enpair_func(builder, x, y),
            lambda x, y: self.depair_func(builder, x, y),
        )
