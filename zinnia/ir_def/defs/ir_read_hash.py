from typing import List, Dict, Optional, Tuple

from zinnia.compile.triplet import Value, IntegerValue
from zinnia.compile.ir.ir_stmt import IRStatement
from zinnia.ir_def.abstract_ir import AbstractIR
from zinnia.debug.dbg_info import DebugInfo


class ReadHashIR(AbstractIR):
    def __init__(self, indices: Tuple[int, ...], is_public: bool):
        super().__init__()
        self.indices = indices
        self.is_public = is_public

    def get_signature(self) -> str:
        return f"read_hash[{self.indices}][{self.is_public}]"

    def __eq__(self, other):
        return super().__eq__(other) and self.indices == other.indices and self.is_public == other.is_public

    def is_fixed_ir(self) -> bool:
        return True

    def build_ir(self, ir_id: int, args: List[Value], dbg: Optional[DebugInfo] = None) -> Tuple[Value, IRStatement]:
        return IntegerValue(None, ir_id), IRStatement(ir_id, self, [], dbg)

    def export(self) -> Dict:
        return {
            "indices": list(self.indices),
            "is_public": self.is_public
        }

    @staticmethod
    def import_from(data: Dict) -> 'ReadHashIR':
        indices = data['indices']
        return ReadHashIR(tuple(indices), data['is_public'])
