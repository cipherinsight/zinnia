from typing import List, Dict, Optional, Any, Tuple

from z3 import z3

from zinnia.compile.triplet import Value, IntegerValue
from zinnia.compile.ir.ir_stmt import IRStatement
from zinnia.config.mock_exec_config import MockExecConfig
from zinnia.ir_def.abstract_ir import AbstractIR
from zinnia.debug.dbg_info import DebugInfo


class SignIIR(AbstractIR):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "sign_i"

    def infer(self, args: List[Value], dbg: Optional[DebugInfo] = None) -> Any:
        x = args[0]
        assert isinstance(x, IntegerValue)
        return (1 if x.c_val() > 0 else (-1 if x.c_val() < 0 else 0)) if x.c_val() is not None else None

    def mock_exec(self, args: List[Any], config: MockExecConfig) -> Any:
        return int(1 if args[0] > 0 else (-1 if args[0] < 0 else 0))

    def build_ir(self, ir_id: int, args: List[Value], dbg: Optional[DebugInfo] = None) -> Tuple[Value, IRStatement]:
        x = args[0]
        assert isinstance(x, IntegerValue)
        return IntegerValue(
            self.infer(args, dbg), ir_id,
            z3e=z3.If(x.z3_sym > 0, 1, z3.If(x.z3_sym < 0, -1, 0)), rel=x.z3_rel
        ), IRStatement(ir_id, self, [x.ptr()], dbg)

    def export(self) -> Dict:
        return {}

    @staticmethod
    def import_from(data: Dict) -> 'SignIIR':
        return SignIIR()
