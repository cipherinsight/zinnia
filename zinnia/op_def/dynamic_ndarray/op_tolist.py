from typing import List, Optional, Sequence

from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.compile.triplet import DynamicNDArrayValue, ListValue, NumberValue, Value
from zinnia.debug.dbg_info import DebugInfo
from zinnia.op_def.abstract.abstract_op import AbstractOp
from zinnia.op_def.dynamic_ndarray.view_utils import flatten_logical_values, logical_num_elements


class DynamicNDArray_ToListOp(AbstractOp):
    def get_signature(self) -> str:
        return "DynamicNDArray.tolist"

    @classmethod
    def get_name(cls) -> str:
        return "tolist"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [AbstractOp._ParamEntry("self")]

    @staticmethod
    def _build_nested(values: Sequence[NumberValue], shape: tuple[int, ...]):
        if len(shape) == 1:
            return list(values[: shape[0]])
        step = 1
        for dim in shape[1:]:
            step *= dim
        out = []
        for i in range(shape[0]):
            out.append(DynamicNDArray_ToListOp._build_nested(values[i * step: (i + 1) * step], shape[1:]))
        return out

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        the_self = kwargs["self"]
        assert isinstance(the_self, DynamicNDArrayValue)

        shape = the_self.logical_shape()
        if len(shape) == 0:
            return builder.op_square_brackets([])

        flat_values = flatten_logical_values(builder, the_self)
        total = logical_num_elements(shape)
        flat_values = flat_values[:total]
        nested = self._build_nested(flat_values, shape)

        def _recursive_list_builder(_depth, _list) -> ListValue:
            if _depth == 1:
                return builder.op_square_brackets([value for value in _list])
            return builder.op_square_brackets([_recursive_list_builder(_depth - 1, value) for value in _list])

        return _recursive_list_builder(len(shape), nested)
