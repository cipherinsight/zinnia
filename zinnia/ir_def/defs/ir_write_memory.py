from typing import List, Dict, Optional, Tuple

from zinnia.compile.triplet import Value, IntegerValue, NoneValue
from zinnia.compile.ir.ir_stmt import IRStatement
from zinnia.ir_def.abstract_ir import AbstractIR
from zinnia.debug.dbg_info import DebugInfo


class WriteMemoryIR(AbstractIR):
    def __init__(self, segment_id: int):
        super().__init__()
        self.segment_id = segment_id

    def get_signature(self) -> str:
        return f"write_memory[{self.segment_id}]"

    def __eq__(self, other):
        return super().__eq__(other) and self.segment_id == other.segment_id

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
        }

    @staticmethod
    def import_from(data: Dict) -> 'WriteMemoryIR':
        return WriteMemoryIR(segment_id=data["segment_id"])
