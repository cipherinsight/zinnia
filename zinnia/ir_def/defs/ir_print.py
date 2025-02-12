from typing import List, Dict, Optional, Any, Tuple

from zinnia.compile.triplet import Value, StringValue, NoneValue, IntegerValue
from zinnia.compile.ir.ir_stmt import IRStatement
from zinnia.config.mock_exec_config import MockExecConfig
from zinnia.ir_def.abstract_ir import AbstractIR
from zinnia.debug.dbg_info import DebugInfo


class PrintIR(AbstractIR):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "print"

    def is_fixed_ir(self) -> bool:
        return True

    def infer(self, args: List[Value], dbg: Optional[DebugInfo] = None) -> Any:
        return None

    def mock_exec(self, args: List[Any], config: MockExecConfig) -> Any:
        return None

    def build_ir(self, ir_id: int, args: List[Value], dbg: Optional[DebugInfo] = None) -> Tuple[Value, IRStatement]:
        cond, x = args[0], args[1]
        assert isinstance(cond, IntegerValue)
        assert isinstance(x, StringValue)
        return NoneValue(), IRStatement(ir_id, self, [cond.ptr(), x.ptr()], dbg)

    def export(self) -> Dict:
        return {}

    @staticmethod
    def import_from(data: Dict) -> 'PrintIR':
        return PrintIR()
