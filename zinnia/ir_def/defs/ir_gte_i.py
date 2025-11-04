from typing import List, Dict, Optional, Any, Tuple

from zinnia.compile.triplet import Value, IntegerValue
from zinnia.compile.ir.ir_stmt import IRStatement
from zinnia.compile.triplet.value.boolean import BooleanValue
from zinnia.config.mock_exec_config import MockExecConfig
from zinnia.ir_def.abstract_ir import AbstractIR
from zinnia.debug.dbg_info import DebugInfo


class GreaterThanOrEqualIIR(AbstractIR):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "gte_i"

    def infer(self, args: List[Value], dbg: Optional[DebugInfo] = None) -> Any:
        lhs, rhs = args[0], args[1]
        assert isinstance(lhs, IntegerValue) and isinstance(rhs, IntegerValue)
        return (True if lhs.val() >= rhs.val() else False) if lhs.val() is not None and rhs.val() is not None else None

    def mock_exec(self, args: List[Any], config: MockExecConfig) -> Any:
        return True if args[0] >= args[1] else False

    def build_ir(self, ir_id: int, args: List[Value], dbg: Optional[DebugInfo] = None) -> Tuple[Value, IRStatement]:
        lhs, rhs = args[0], args[1]
        assert isinstance(lhs, IntegerValue) and isinstance(rhs, IntegerValue)
        return BooleanValue(
            self.infer(args, dbg), ir_id,
        ), IRStatement(ir_id, self, [lhs.ptr(), rhs.ptr()], dbg)

    def export(self) -> Dict:
        return {}

    @staticmethod
    def import_from(data: Dict) -> 'GreaterThanOrEqualIIR':
        return GreaterThanOrEqualIIR()
