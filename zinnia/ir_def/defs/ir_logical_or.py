from typing import List, Dict, Optional, Any, Tuple

from zinnia.compile.triplet import Value

from zinnia.compile.ir.ir_stmt import IRStatement
from zinnia.compile.triplet.value.boolean import BooleanValue
from zinnia.config.mock_exec_config import MockExecConfig
from zinnia.ir_def.abstract_ir import AbstractIR
from zinnia.debug.dbg_info import DebugInfo


class LogicalOrIR(AbstractIR):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "logical_or"

    def infer(self, args: List[Value], dbg: Optional[DebugInfo] = None) -> Any:
        lhs, rhs = args[0], args[1]
        assert isinstance(lhs, BooleanValue) and isinstance(rhs, BooleanValue)
        if lhs.val() is not None and rhs.val() is not None:
            return True if (lhs.val() != False) or (rhs.val() != False) else False
        elif lhs.val() is None and rhs.val() is None:
            return None
        elif lhs.val() is None and rhs.val() is not None:
            return None if (rhs.val() == False) else True
        elif lhs.val() is not None and rhs.val() is None:
            return None if (lhs.val() == False) else True
        raise NotImplementedError()

    def mock_exec(self, args: List[Any], config: MockExecConfig) -> Any:
        return True if (args[0] != False) or (args[1] != False) else False

    def build_ir(self, ir_id: int, args: List[Value], dbg: Optional[DebugInfo] = None) -> Tuple[Value, IRStatement]:
        lhs, rhs = args[0], args[1]
        assert isinstance(lhs, BooleanValue) and isinstance(rhs, BooleanValue)
        return BooleanValue(self.infer(args, dbg), ir_id), IRStatement(ir_id, self, [lhs.ptr(), rhs.ptr()], dbg)

    def export(self) -> Dict:
        return {}

    @staticmethod
    def import_from(data: Dict) -> 'LogicalOrIR':
        return LogicalOrIR()
