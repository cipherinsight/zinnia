from typing import Dict, List, Optional, Tuple

from zinnia.compile.ir.ir_stmt import IRStatement
from zinnia.compile.triplet import IntegerValue, NoneValue, Value
from zinnia.debug.dbg_info import DebugInfo
from zinnia.ir_def.abstract_ir import AbstractIR


class DynamicNDArraySetItemIR(AbstractIR):
    def __init__(self, array_id: int, segment_id: int):
        super().__init__()
        self.array_id = array_id
        self.segment_id = segment_id

    def get_signature(self) -> str:
        return f"dynamic_ndarray_set_item[{self.array_id}][{self.segment_id}]"

    def __eq__(self, other):
        return (
            super().__eq__(other)
            and self.array_id == other.array_id
            and self.segment_id == other.segment_id
        )

    def is_fixed_ir(self) -> bool:
        return True

    def build_ir(self, ir_id: int, args: List[Value], dbg: Optional[DebugInfo] = None) -> Tuple[Value, IRStatement]:
        assert len(args) == 2
        assert isinstance(args[0], IntegerValue)
        assert isinstance(args[1], IntegerValue)
        ptr_0 = args[0].ptr()
        ptr_1 = args[1].ptr()
        assert ptr_0 is not None and ptr_1 is not None
        return NoneValue(), IRStatement(ir_id, self, [ptr_0, ptr_1], dbg)

    def export(self) -> Dict:
        return {
            "array_id": self.array_id,
            "segment_id": self.segment_id,
        }

    @staticmethod
    def import_from(data: Dict) -> "DynamicNDArraySetItemIR":
        return DynamicNDArraySetItemIR(
            array_id=data["array_id"],
            segment_id=data["segment_id"],
        )
