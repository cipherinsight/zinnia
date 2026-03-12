from typing import Dict, List, Optional, Tuple

from zinnia.compile.ir.ir_stmt import IRStatement
from zinnia.compile.triplet import NoneValue, Value
from zinnia.debug.dbg_info import DebugInfo
from zinnia.ir_def.abstract_ir import AbstractIR


class AllocateDynamicNDArrayMetaIR(AbstractIR):
    def __init__(self, array_id: int, dtype_name: str, max_length: int, max_rank: int):
        super().__init__()
        self.array_id = array_id
        self.dtype_name = dtype_name
        self.max_length = max_length
        self.max_rank = max_rank

    def get_signature(self) -> str:
        return (
            f"alloc_dynamic_ndarray_meta[{self.array_id}]"
            f"[{self.dtype_name}][{self.max_length}][{self.max_rank}]"
        )

    def __eq__(self, other):
        return (
            super().__eq__(other)
            and self.array_id == other.array_id
            and self.dtype_name == other.dtype_name
            and self.max_length == other.max_length
            and self.max_rank == other.max_rank
        )

    def is_fixed_ir(self) -> bool:
        return True

    def build_ir(self, ir_id: int, args: List[Value], dbg: Optional[DebugInfo] = None) -> Tuple[Value, IRStatement]:
        return NoneValue(), IRStatement(ir_id, self, [], dbg)

    def export(self) -> Dict:
        return {
            "array_id": self.array_id,
            "dtype_name": self.dtype_name,
            "max_length": self.max_length,
            "max_rank": self.max_rank,
        }

    @staticmethod
    def import_from(data: Dict) -> "AllocateDynamicNDArrayMetaIR":
        return AllocateDynamicNDArrayMetaIR(
            array_id=data["array_id"],
            dtype_name=data["dtype_name"],
            max_length=data["max_length"],
            max_rank=data["max_rank"],
        )
