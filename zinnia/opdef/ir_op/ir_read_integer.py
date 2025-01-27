from typing import List, Dict, Optional, Tuple

from zinnia.compile.builder.value import Value, IntegerValue
from zinnia.compile.ir.ir_stmt import IRStatement
from zinnia.opdef.ir_op.abstract_ir import AbstractIR
from zinnia.opdef.nocls.abstract_op import AbstractOp
from zinnia.debug.dbg_info import DebugInfo


class ReadIntegerIR(AbstractIR):
    def __init__(self, indices: Tuple[int, ...]):
        super().__init__()
        self.indices = indices

    def get_signature(self) -> str:
        return f"read_integer[{', '.join(map(str, self.indices))}]"

    @classmethod
    def get_name(cls) -> str:
        return "read_integer"

    def __eq__(self, other):
        return super().__eq__(other) and self.indices == other.indices

    def is_fixed_ir(self) -> bool:
        return True

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return []

    def build_ir(self, ir_id: int, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Tuple[Value, IRStatement]:
        return IntegerValue(None, ir_id), IRStatement(ir_id, self, [], dbg)

    def export(self) -> Dict:
        return {
            "indices": list(self.indices)
        }

    @staticmethod
    def import_from(data: Dict) -> 'ReadIntegerIR':
        return ReadIntegerIR(tuple(data['indices']))
