from typing import List, Dict, Optional, Any, Tuple

from zinnia.compile.triplet import Value, IntegerValue
from zinnia.compile.ir.ir_stmt import IRStatement
from zinnia.config.mock_exec_config import MockExecConfig
from zinnia.ir_def.abstract_ir import AbstractIR
from zinnia.debug.dbg_info import DebugInfo


class MulIIR(AbstractIR):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "mul_i"

    def infer(self, args: List[Value], dbg: Optional[DebugInfo] = None) -> Any:
        lhs, rhs = args[0], args[1]
        assert isinstance(lhs, IntegerValue) and isinstance(rhs, IntegerValue)
        return lhs.c_val() * rhs.c_val() if lhs.c_val() is not None and rhs.c_val() is not None else None

    def mock_exec(self, args: List[Any], config: MockExecConfig) -> Any:
        return int(int(args[0]) * int(args[1]))

    def build_ir(self, ir_id: int, args: List[Value], dbg: Optional[DebugInfo] = None) -> Tuple[Value, IRStatement]:
        lhs, rhs = args[0], args[1]
        assert isinstance(lhs, IntegerValue) and isinstance(rhs, IntegerValue)
        return IntegerValue(
            self.infer(args, dbg), ir_id,
            z3e=lhs.z3_sym * rhs.z3_sym, rel=lhs.z3_rel + rhs.z3_rel
        ), IRStatement(ir_id, self, [lhs.ptr(), rhs.ptr()], dbg)

    def export(self) -> Dict:
        return {}

    @staticmethod
    def import_from(data: Dict) -> 'MulIIR':
        return MulIIR()
