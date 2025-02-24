from typing import List, Dict, Optional, Any, Tuple

from zinnia.compile.triplet import Value, IntegerValue, FloatValue
from zinnia.compile.ir.ir_stmt import IRStatement
from zinnia.config.mock_exec_config import MockExecConfig
from zinnia.ir_def.abstract_ir import AbstractIR
from zinnia.debug.dbg_info import DebugInfo


class PoseidonHashIR(AbstractIR):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "poseidon_hash"

    def infer(self, args: List[Value], dbg: Optional[DebugInfo] = None) -> Any:
        return None

    def mock_exec(self, args: List[Any], config: MockExecConfig) -> Any:
        # TODO: implement mock execution for poseidon hash
        return 0

    def build_ir(self, ir_id: int, args: List[Value], dbg: Optional[DebugInfo] = None) -> Tuple[Value, IRStatement]:
        assert all([isinstance(arg, IntegerValue) or isinstance(arg, FloatValue) for arg in args])
        return IntegerValue(self.infer(args, dbg), ir_id), IRStatement(ir_id, self, [arg.ptr() for arg in args], dbg)

    def export(self) -> Dict:
        return {}

    @staticmethod
    def import_from(data: Dict) -> 'PoseidonHashIR':
        return PoseidonHashIR()
