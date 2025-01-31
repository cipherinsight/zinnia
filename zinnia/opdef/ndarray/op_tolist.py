from typing import List, Dict, Optional

from zinnia.opdef.abstract.abstract_op import AbstractOp
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.builder.abstract_ir_builder import AbsIRBuilderInterface
from zinnia.compile.builder.value import NDArrayValue, Value, ListValue


class NDArray_ToListOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "NDArray.tolist"

    @classmethod
    def get_name(cls) -> str:
        return "tolist"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("self")
        ]

    def build(self, builder: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        the_self = kwargs["self"]
        assert isinstance(the_self, NDArrayValue)
        the_list = the_self.tolist()
        def _recursive_list_builder(_depth, _list) -> ListValue:
            if _depth == 1:
                return builder.op_square_brackets([value for value in _list])
            return builder.op_square_brackets([_recursive_list_builder(_depth - 1, value) for value in _list])
        return _recursive_list_builder(len(the_self.shape()), the_list)
