from typing import List, Dict, Optional, Any, Tuple

from zinnia.compile.ir.ir_stmt import IRStatement
from zinnia.config.mock_exec_config import MockExecConfig
from zinnia.ir_def.abstract_ir import AbstractIR
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.triplet import Value, FloatValue


class ConstantFloatIR(AbstractIR):
    def __init__(self, value: float):
        super().__init__()
        assert value is not None
        self.value = value

    def get_signature(self) -> str:
        return f"constant_float[{self.value}]"

    def __eq__(self, other):
        return super().__eq__(other) and self.value == other.value

    def infer(self, args: List[Value], dbg: Optional[DebugInfo] = None) -> Any:
        return self.value

    def mock_exec(self, args: List[Any], config: MockExecConfig) -> Any:
        return float(self.value)

    def build_ir(self, ir_id: int, args: List[Value], dbg: Optional[DebugInfo] = None) -> Tuple[Value, IRStatement]:
        return FloatValue(self.value, ir_id), IRStatement(ir_id, self, [], dbg)

    def export(self) -> Dict:
        return {
            "value": self.value
        }

    @staticmethod
    def import_from(data: Dict) -> 'ConstantFloatIR':
        return ConstantFloatIR(data['value'])
