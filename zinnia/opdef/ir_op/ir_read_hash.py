from typing import List, Dict, Optional, Tuple

from zinnia.compile.builder.value import Value, IntegerValue
from zinnia.compile.ir.ir_stmt import IRStatement
from zinnia.opdef.ir_op.abstract_ir import AbstractIR
from zinnia.opdef.nocls.abstract_op import AbstractOp
from zinnia.debug.dbg_info import DebugInfo


class ReadHashIR(AbstractIR):
    def __init__(self, major: int, minor: int):
        super().__init__()
        self.major = major
        self.minor = minor

    def get_signature(self) -> str:
        return f"read_hash[{self.major}, {self.minor}]"

    @classmethod
    def get_name(cls) -> str:
        return "read_hash"

    def __eq__(self, other):
        return super().__eq__(other) and self.major == other.major and self.minor == other.minor

    def is_fixed_ir(self) -> bool:
        return True

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return []

    def build_ir(self, ir_id: int, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Tuple[Value, IRStatement]:
        return IntegerValue(None, ir_id), IRStatement(ir_id, self, [], dbg)

    def export(self) -> Dict:
        return {
            "indices": (self.major, self.minor)
        }

    @staticmethod
    def import_from(data: Dict) -> 'ReadHashIR':
        indices = data['indices']
        return ReadHashIR(indices[0], indices[1])
