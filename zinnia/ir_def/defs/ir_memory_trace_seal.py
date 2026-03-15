from typing import List, Dict, Optional, Tuple

from zinnia.compile.triplet import Value, NoneValue
from zinnia.compile.ir.ir_stmt import IRStatement
from zinnia.ir_def.abstract_ir import AbstractIR
from zinnia.debug.dbg_info import DebugInfo


class MemoryTraceSealIR(AbstractIR):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "memory_trace_seal"

    def is_fixed_ir(self) -> bool:
        return True

    def build_ir(self, ir_id: int, args: List[Value], dbg: Optional[DebugInfo] = None) -> Tuple[Value, IRStatement]:
        return NoneValue(), IRStatement(ir_id, self, [], dbg)

    def export(self) -> Dict:
        return {}

    @staticmethod
    def import_from(data: Dict) -> 'MemoryTraceSealIR':
        return MemoryTraceSealIR()
