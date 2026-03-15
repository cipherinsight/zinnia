from typing import List, Dict, Optional, Tuple

from zinnia.compile.triplet import Value, NoneValue
from zinnia.compile.ir.ir_stmt import IRStatement
from zinnia.ir_def.abstract_ir import AbstractIR
from zinnia.debug.dbg_info import DebugInfo


class AllocateMemoryIR(AbstractIR):
    def __init__(self, segment_id: int, size: int, init_value: int = 0):
        super().__init__()
        self.segment_id = segment_id
        self.size = size
        self.init_value = init_value

    def get_signature(self) -> str:
        return f"allocate_memory[{self.segment_id}][{self.size}][{self.init_value}]"

    def __eq__(self, other):
        return (
            super().__eq__(other)
            and self.segment_id == other.segment_id
            and self.size == other.size
            and self.init_value == other.init_value
        )

    def is_fixed_ir(self) -> bool:
        return True

    def build_ir(self, ir_id: int, args: List[Value], dbg: Optional[DebugInfo] = None) -> Tuple[Value, IRStatement]:
        return NoneValue(), IRStatement(ir_id, self, [], dbg)

    def export(self) -> Dict:
        return {
            "segment_id": self.segment_id,
            "size": self.size,
            "init_value": self.init_value,
        }

    @staticmethod
    def import_from(data: Dict) -> 'AllocateMemoryIR':
        return AllocateMemoryIR(
            segment_id=data["segment_id"],
            size=data["size"],
            init_value=data.get("init_value", 0),
        )
