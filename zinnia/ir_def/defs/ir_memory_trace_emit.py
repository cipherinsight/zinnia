from typing import List, Dict, Optional, Tuple

from zinnia.compile.triplet import Value, IntegerValue, NoneValue
from zinnia.compile.ir.ir_stmt import IRStatement
from zinnia.ir_def.abstract_ir import AbstractIR
from zinnia.debug.dbg_info import DebugInfo


class MemoryTraceEmitIR(AbstractIR):
    def __init__(self, segment_id: int, is_write: bool):
        super().__init__()
        self.segment_id = segment_id
        self.is_write = is_write

    def get_signature(self) -> str:
        return f"memory_trace_emit[{self.segment_id}][{self.is_write}]"

    def __eq__(self, other):
        return (
            super().__eq__(other)
            and self.segment_id == other.segment_id
            and self.is_write == other.is_write
        )

    def is_fixed_ir(self) -> bool:
        return True

    def build_ir(self, ir_id: int, args: List[Value], dbg: Optional[DebugInfo] = None) -> Tuple[Value, IRStatement]:
        assert len(args) == 2
        assert isinstance(args[0], IntegerValue)
        assert isinstance(args[1], IntegerValue)
        return NoneValue(), IRStatement(ir_id, self, [args[0].ptr(), args[1].ptr()], dbg)

    def export(self) -> Dict:
        return {
            "segment_id": self.segment_id,
            "is_write": self.is_write,
        }

    @staticmethod
    def import_from(data: Dict) -> 'MemoryTraceEmitIR':
        return MemoryTraceEmitIR(segment_id=data["segment_id"], is_write=data["is_write"])
