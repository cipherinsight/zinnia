from typing import List, Dict, Optional, Any, Tuple

from zinnia.compile.triplet import Value, IntegerValue, FloatValue
from zinnia.compile.ir.ir_stmt import IRStatement
from zinnia.compile.triplet.value.boolean import BooleanValue
from zinnia.config.mock_exec_config import MockExecConfig
from zinnia.ir_def.abstract_ir import AbstractIR
from zinnia.debug.dbg_info import DebugInfo


class SelectFIR(AbstractIR):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "select_f"

    def infer(self, args: List[Value], dbg: Optional[DebugInfo] = None) -> Any:
        cond, tv, fv = args[0], args[1], args[2]
        assert isinstance(cond, BooleanValue) and isinstance(tv, FloatValue) and isinstance(fv, FloatValue)
        if cond.c_val() is None:
            return None
        elif cond.c_val() != False:
            return tv.c_val()
        else:
            return fv.c_val()

    def mock_exec(self, args: List[Any], config: MockExecConfig) -> Any:
        return float(args[1] if args[0] != 0 else args[2])

    def build_ir(self, ir_id: int, args: List[Value], dbg: Optional[DebugInfo] = None) -> Tuple[Value, IRStatement]:
        cond, tv, fv = args[0], args[1], args[2]
        assert isinstance(cond, BooleanValue) and isinstance(tv, FloatValue) and isinstance(fv, FloatValue)
        return FloatValue(self.infer(args, dbg), ir_id), IRStatement(ir_id, self, [cond.ptr(), tv.ptr(), fv.ptr()], dbg)

    def export(self) -> Dict:
        return {}

    @staticmethod
    def import_from(data: Dict) -> 'SelectFIR':
        return SelectFIR()
