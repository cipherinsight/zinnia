from typing import List, Dict, Optional, Any, Tuple

import math

from zinnia.compile.triplet import Value, FloatValue
from zinnia.compile.ir.ir_stmt import IRStatement
from zinnia.config.mock_exec_config import MockExecConfig
from zinnia.ir_def.abstract_ir import AbstractIR
from zinnia.debug.dbg_info import DebugInfo


class TanHFIR(AbstractIR):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "tanh_f"

    def infer(self, args: List[Value], dbg: Optional[DebugInfo] = None) -> Any:
        x = args[0]
        assert isinstance(x, FloatValue)
        return math.tanh(x.val()) if x.val() is not None else None

    def mock_exec(self, args: List[Any], config: MockExecConfig) -> Any:
        return float(math.tanh(args[0]))

    def build_ir(self, ir_id: int, args: List[Value], dbg: Optional[DebugInfo] = None) -> Tuple[Value, IRStatement]:
        x = args[0]
        assert isinstance(x, FloatValue)
        return FloatValue(self.infer(args, dbg), ir_id), IRStatement(ir_id, self, [x.ptr()], dbg)

    def export(self) -> Dict:
        return {}

    @staticmethod
    def import_from(data: Dict) -> 'TanHFIR':
        return TanHFIR()
