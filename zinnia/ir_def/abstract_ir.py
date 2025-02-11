from typing import List, Optional, Tuple, Any, Dict

from zinnia.compile.triplet import Value
from zinnia.config.mock_exec_config import MockExecConfig
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.ir.ir_stmt import IRStatement


class AbstractIR:
    def __init__(self):
        super().__init__()

    def __eq__(self, other):
        return self.__class__ == other.__class__

    def get_signature(self) -> str:
        raise NotImplementedError()

    def is_fixed_ir(self) -> bool:
        return False

    def build_ir(self, ir_id: int, args: List[Value], dbg: Optional[DebugInfo] = None) -> Tuple[Value, IRStatement]:
        raise NotImplementedError()

    def infer(self, args: List[Value], dbg: Optional[DebugInfo] = None) -> Any:
        raise NotImplementedError()

    def mock_exec(self, args: List[Any], config: MockExecConfig) -> Any:
        raise NotImplementedError()

    def export(self) -> Dict:
        raise NotImplementedError()

    @staticmethod
    def import_from(data: Dict) -> 'AbstractIR':
        raise NotImplementedError()
