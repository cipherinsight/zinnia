from typing import Dict, List, Optional, Tuple

from zinnia.compile.ir.ir_stmt import IRStatement
from zinnia.compile.triplet import IntegerValue, NoneValue, Value
from zinnia.debug.dbg_info import DebugInfo
from zinnia.ir_def.abstract_ir import AbstractIR


class WitnessDynamicNDArrayMetaIR(AbstractIR):
    def __init__(self, array_id: int, max_rank: int):
        super().__init__()
        self.array_id = array_id
        self.max_rank = max_rank

    def get_signature(self) -> str:
        return f"witness_dynamic_ndarray_meta[{self.array_id}][{self.max_rank}]"

    def __eq__(self, other):
        return (
            super().__eq__(other)
            and self.array_id == other.array_id
            and self.max_rank == other.max_rank
        )

    def is_fixed_ir(self) -> bool:
        return True

    def build_ir(self, ir_id: int, args: List[Value], dbg: Optional[DebugInfo] = None) -> Tuple[Value, IRStatement]:
        expected_len = 2 + 2 * self.max_rank
        assert len(args) == expected_len
        ptrs = []
        for arg in args:
            assert isinstance(arg, IntegerValue)
            ptrs.append(arg.ptr())
        return NoneValue(), IRStatement(ir_id, self, ptrs, dbg)

    def export(self) -> Dict:
        return {
            "array_id": self.array_id,
            "max_rank": self.max_rank,
        }

    @staticmethod
    def import_from(data: Dict) -> "WitnessDynamicNDArrayMetaIR":
        return WitnessDynamicNDArrayMetaIR(
            array_id=data["array_id"],
            max_rank=data["max_rank"],
        )
