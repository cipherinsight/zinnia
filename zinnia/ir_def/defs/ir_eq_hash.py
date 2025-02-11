from typing import List, Dict, Optional, Any, Tuple

from zinnia.compile.triplet import Value, IntegerValue
from zinnia.compile.ir.ir_stmt import IRStatement
from zinnia.config.mock_exec_config import MockExecConfig
from zinnia.ir_def.abstract_ir import AbstractIR
from zinnia.debug.dbg_info import DebugInfo


class EqualHashIR(AbstractIR):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "eq_hash"

    def infer(self, args: List[Value], dbg: Optional[DebugInfo] = None) -> Any:
        lhs, rhs = args[0], args[1]
        assert isinstance(lhs, IntegerValue) and isinstance(rhs, IntegerValue)
        return 1

    def mock_exec(self, args: List[Any], config: MockExecConfig) -> Any:
        # TODO: implement hash equal test
        return 1

    def build_ir(self, ir_id: int, args: List[Value], dbg: Optional[DebugInfo] = None) -> Tuple[Value, IRStatement]:
        lhs, rhs = args[0], args[1]
        assert isinstance(lhs, IntegerValue) and isinstance(rhs, IntegerValue)
        return IntegerValue(self.infer(args, dbg), ir_id), IRStatement(ir_id, self, [lhs.ptr(), rhs.ptr()], dbg)

    def export(self) -> Dict:
        return {}

    @staticmethod
    def import_from(data: Dict) -> 'EqualHashIR':
        return EqualHashIR()
