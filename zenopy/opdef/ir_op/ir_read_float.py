from typing import List, Dict, Optional, Tuple

from zenopy.builder.value import Value, FloatValue
from zenopy.ir.ir_stmt import IRStatement
from zenopy.opdef.ir_op.abstract_ir import AbstractIR
from zenopy.opdef.nocls.abstract_op import AbstractOp
from zenopy.debug.dbg_info import DebugInfo


class ReadFloatIR(AbstractIR):
    def __init__(self, major: int, minor: int):
        super().__init__()
        self.major = major
        self.minor = minor

    def get_signature(self) -> str:
        return f"read_float[{self.major}, {self.minor}]"

    @classmethod
    def get_name(cls) -> str:
        return "read_float"

    def __eq__(self, other):
        return super().__eq__(other) and self.major == other.major and self.minor == other.minor

    def is_fixed_ir(self) -> bool:
        return True

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return []

    def build_ir(self, ir_id: int, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Tuple[Value, IRStatement]:
        return FloatValue(None, ir_id), IRStatement(ir_id, self, [], dbg)
