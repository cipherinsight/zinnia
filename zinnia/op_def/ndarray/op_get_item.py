from typing import List, Optional

from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.debug.dbg_info import DebugInfo
from zinnia.op_def.abstract.abstract_ndarray_item_slice import AbstractNDArrayItemSlice
from zinnia.op_def.abstract.abstract_op import AbstractOp
from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import Value, NDArrayValue, DynamicNDArrayValue, ListValue, TupleValue, IntegerValue
from zinnia.op_def.ndarray.op_filter import NDArray_FilterOp
from zinnia.op_def.dynamic_ndarray.op_filter import DynamicNDArray_FilterOp
from zinnia.op_def.dynamic_ndarray.op_get_item import DynamicNDArray_GetItemOp


class NDArray_GetItemOp(AbstractNDArrayItemSlice):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "NDArray.__get_item__"

    @classmethod
    def get_name(cls) -> str:
        return "__get_item__"

    @classmethod
    def requires_condition(cls) -> bool:
        return True

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("self"),
            AbstractOp._ParamEntry("slicing_params")
        ]

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        the_self = kwargs['self']
        raw_slicing_params = kwargs['slicing_params']
        assert isinstance(raw_slicing_params, ListValue)
        assert isinstance(the_self, NDArrayValue)

        # Boolean-mask filtering: arr[mask]
        if len(raw_slicing_params.values()) == 1 and isinstance(raw_slicing_params.values()[0], NDArrayValue):
            mask = raw_slicing_params.values()[0]
            if isinstance(the_self, DynamicNDArrayValue) or isinstance(mask, DynamicNDArrayValue):
                op = DynamicNDArray_FilterOp()
                dyn_self = the_self if isinstance(the_self, DynamicNDArrayValue) else the_self.to_dynamic_ndarray()
                dyn_mask = mask if isinstance(mask, DynamicNDArrayValue) else mask.to_dynamic_ndarray()
                kwargs2 = op.argparse(dbg, [dyn_self, dyn_mask], {})
                return op.build(builder, OpArgsContainer(kwargs2, kwargs.get_condition()), dbg)
            op = NDArray_FilterOp()
            kwargs2 = op.argparse(dbg, [the_self, mask], {})
            return op.build(builder, OpArgsContainer(kwargs2, kwargs.get_condition()), dbg)

        # Dynamic ndarray indexing/slicing is handled by the dedicated dynamic operator.
        if isinstance(the_self, DynamicNDArrayValue):
            op = DynamicNDArray_GetItemOp()
            kwargs2 = op.argparse(dbg, [the_self, raw_slicing_params], {})
            return op.build(builder, OpArgsContainer(kwargs2, kwargs.get_condition()), dbg)

        has_dynamic_slice_component = False
        has_dynamic_scalar_index = False
        for sp in raw_slicing_params.values():
            if isinstance(sp, TupleValue):
                for elem in sp.values():
                    if isinstance(elem, IntegerValue) and elem.val(builder) is None:
                        has_dynamic_slice_component = True
            if isinstance(sp, IntegerValue) and sp.val(builder) is None:
                has_dynamic_scalar_index = True

        if (has_dynamic_slice_component or has_dynamic_scalar_index) and len(raw_slicing_params.values()) == 1 and len(the_self.shape()) == 1:
            op = DynamicNDArray_GetItemOp()
            dyn_self = the_self.to_dynamic_ndarray()
            kwargs2 = op.argparse(dbg, [dyn_self, raw_slicing_params], {})
            return op.build(builder, OpArgsContainer(kwargs2, kwargs.get_condition()), dbg)

        slicing_params = self.check_slicing_params_datatype(builder, raw_slicing_params, dbg)

        self.check_slicing_dimensions(slicing_params.values(), the_self.shape(), dbg)
        candidates, conditions = self.find_all_candidates(builder, slicing_params.values(), the_self.shape(), kwargs.get_condition(), dbg)
        result = the_self.get_item(candidates[0])
        for candidate, condition in zip(candidates[1:], conditions[1:]):
            result = builder.op_select(condition, the_self.get_item(candidate), result)
        return result
