from typing import List, Dict, Optional, Any, Tuple

from z3 import z3

from zinnia.compile.triplet import Value

from zinnia.compile.ir.ir_stmt import IRStatement
from zinnia.compile.triplet.value.boolean import BooleanValue
from zinnia.config.mock_exec_config import MockExecConfig
from zinnia.ir_def.abstract_ir import AbstractIR
from zinnia.debug.dbg_info import DebugInfo


class LogicalAndIR(AbstractIR):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "logical_and"

    def infer(self, args: List[Value], dbg: Optional[DebugInfo] = None) -> Any:
        lhs, rhs = args[0], args[1]
        assert isinstance(lhs, BooleanValue) and isinstance(rhs, BooleanValue)
        if lhs.c_val() is not None and rhs.c_val() is not None:
            return True if (lhs.c_val() != False) and (rhs.c_val() != False) else False
        elif lhs.c_val() is None and rhs.c_val() is None:
            return None
        elif lhs.c_val() is None and rhs.c_val() is not None:
            return None if (rhs.c_val() != False) else False
        elif lhs.c_val() is not None and rhs.c_val() is None:
            return None if (lhs.c_val() != False) else False
        raise NotImplementedError()

    def mock_exec(self, args: List[Any], config: MockExecConfig) -> Any:
        return True if (args[0] != False) and (args[1] != False) else False

    def build_ir(self, ir_id: int, args: List[Value], dbg: Optional[DebugInfo] = None) -> Tuple[Value, IRStatement]:
        lhs, rhs = args[0], args[1]
        assert isinstance(lhs, BooleanValue) and isinstance(rhs, BooleanValue)
        return BooleanValue(
            self.infer(args, dbg), ir_id,
            z3e=z3.And(lhs.z3_sym, rhs.z3_sym), rel=lhs.z3_rel + rhs.z3_rel
        ), IRStatement(ir_id, self, [lhs.ptr(), rhs.ptr()], dbg)

    def export(self) -> Dict:
        return {}

    @staticmethod
    def import_from(data: Dict) -> 'LogicalAndIR':
        return LogicalAndIR()
