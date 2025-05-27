from typing import List, Dict, Optional, Any, Tuple

from zinnia.compile.triplet import Value, IntegerValue, NoneValue
from zinnia.compile.ir.ir_stmt import IRStatement
from zinnia.ir_def.abstract_ir import AbstractIR
from zinnia.debug.dbg_info import DebugInfo


class AssertIR(AbstractIR):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "assert"

    def is_fixed_ir(self) -> bool:
        return True

    def infer(self, args: List[Value], dbg: Optional[DebugInfo] = None) -> Any:
        test = args[0]
        assert isinstance(test, IntegerValue)
        return None

    def build_ir(self, ir_id: int, args: List[Value], dbg: Optional[DebugInfo] = None) -> Tuple[Value, IRStatement]:
        test = args[0]
        assert isinstance(test, IntegerValue)
        return NoneValue(), IRStatement(ir_id, self, [test.ptr()], dbg)

    def export(self) -> Dict:
        return {}

    @staticmethod
    def import_from(data: Dict) -> 'AssertIR':
        return AssertIR()
