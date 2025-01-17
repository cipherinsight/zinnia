from typing import List, Dict, Optional, Tuple

from pyzk.builder.abstract_ir_builder import AbsIRBuilderInterface
from pyzk.builder.value import Value, IntegerValue
from pyzk.ir.ir_stmt import IRStatement

from pyzk.opdef.ir_op.abstract_ir import AbstractIR
from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.debug.dbg_info import DebugInfo


class ReadIntegerIR(AbstractIR):
    def __init__(self, major: int, minor: int):
        super().__init__()
        self.major = major
        self.minor = minor

    def get_signature(self) -> str:
        return f"read_integer[{self.major}, {self.minor}]"

    @classmethod
    def get_name(cls) -> str:
        return "read_integer"

    def __eq__(self, other):
        return super().__eq__(other) and self.major == other.major and self.minor == other.minor

    def is_fixed_ir(self) -> bool:
        return True

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return []

    def build_ir(self, ir_id: int, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Tuple[Value, IRStatement]:
        return IntegerValue(None, ir_id), IRStatement(ir_id, self, [], dbg)
