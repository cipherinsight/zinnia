from typing import List, Dict, Optional, Any, Tuple

from zinnia.compile.builder.value import Value, IntegerValue, NoneValue
from zinnia.compile.ir.ir_stmt import IRStatement
from zinnia.config.mock_exec_config import MockExecConfig
from zinnia.opdef.ir_op.abstract_ir import AbstractIR
from zinnia.opdef.abstract.abstract_op import AbstractOp
from zinnia.debug.dbg_info import DebugInfo


class ExposePublicIIR(AbstractIR):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "expose_public_i"

    @classmethod
    def get_name(cls) -> str:
        return "expose_public_i"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("x")
        ]

    def is_fixed_ir(self) -> bool:
        return True

    def infer(self, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Any:
        return None

    def mock_exec(self, kwargs: Dict[str, Any], config: MockExecConfig) -> Any:
        return None

    def build_ir(self, ir_id: int, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Tuple[Value, IRStatement]:
        x = kwargs["x"]
        assert isinstance(x, IntegerValue)
        return NoneValue(), IRStatement(ir_id, self, [x.ptr()], dbg)

    def export(self) -> Dict:
        return {}

    @staticmethod
    def import_from(data: Dict) -> 'ExposePublicIIR':
        return ExposePublicIIR()
