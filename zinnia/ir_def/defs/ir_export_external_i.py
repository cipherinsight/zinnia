from typing import List, Dict, Optional, Tuple

from zinnia.compile.triplet import Value, NoneValue, IntegerValue
from zinnia.compile.ir.ir_stmt import IRStatement
from zinnia.ir_def.abstract_ir import AbstractIR
from zinnia.debug.dbg_info import DebugInfo


class ExportExternalIIR(AbstractIR):
    def __init__(self, for_which: int, key: int | str, indices: Tuple[int, ...]):
        super().__init__()
        self.for_which = for_which
        self.indices = indices
        self.key = key

    def get_signature(self) -> str:
        return f"export_external_i[{self.for_which}][{self.key}][{', '.join(map(str, self.indices))}]"

    def __eq__(self, other):
        return super().__eq__(other) and self.indices == other.indices and self.key == other.key and self.for_which == other.for_which

    def is_fixed_ir(self) -> bool:
        return True

    def build_ir(self, ir_id: int, args: List[Value], dbg: Optional[DebugInfo] = None) -> Tuple[Value, IRStatement]:
        assert isinstance(args[0], IntegerValue)
        return NoneValue(), IRStatement(ir_id, self, [args[0].ptr()], dbg)

    def export(self) -> Dict:
        return {
            "for_which": self.for_which,
            "key": self.key,
            "indices": self.indices
        }

    @staticmethod
    def import_from(data: Dict) -> 'ExportExternalIIR':
        return ExportExternalIIR(
            data["for_which"],
            data["key"],
            tuple(data["indices"])
        )
