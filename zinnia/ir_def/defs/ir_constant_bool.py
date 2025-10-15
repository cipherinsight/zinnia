from typing import List, Dict, Optional, Any, Tuple

from zinnia.compile.triplet.value.boolean import BooleanValue
from zinnia.config.mock_exec_config import MockExecConfig
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.ir.ir_stmt import IRStatement
from zinnia.ir_def.abstract_ir import AbstractIR
from zinnia.compile.triplet import Value


class ConstantBoolIR(AbstractIR):
    def __init__(self, value: bool):
        super().__init__()
        assert value is not None
        self.value = value

    def get_signature(self) -> str:
        return f"constant_bool[{self.value}]"

    def __eq__(self, other):
        return super().__eq__(other) and self.value == other.value

    def infer(self, args: List[Value], dbg: Optional[DebugInfo] = None) -> Any:
        return self.value

    def mock_exec(self, args: List[Any], config: MockExecConfig) -> Any:
        return bool(self.value)

    def build_ir(self, ir_id: int, args: List[Value], dbg: Optional[DebugInfo] = None) -> Tuple[Value, IRStatement]:
        return BooleanValue(self.value, ir_id, z3e=self.value), IRStatement(ir_id, self, [], dbg)

    def export(self) -> Dict:
        return {
            "value": self.value
        }

    @staticmethod
    def import_from(data: Dict) -> 'ConstantBoolIR':
        return ConstantBoolIR(data['value'])
