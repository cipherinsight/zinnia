from typing import List, Dict, Optional, Any, Tuple

from zenopy.config.mock_exec_config import MockExecConfig
from zenopy.debug.dbg_info import DebugInfo
from zenopy.compile.ir_stmt import IRStatement
from zenopy.opdef.ir_op.abstract_ir import AbstractIR
from zenopy.opdef.nocls.abstract_op import AbstractOp
from zenopy.builder.abstract_ir_builder import AbsIRBuilderInterface
from zenopy.builder.value import Value, IntegerValue


class ConstantIntIR(AbstractIR):
    def __init__(self, value: int):
        super().__init__()
        assert value is not None
        self.value = value

    def get_signature(self) -> str:
        return f"constant_int[{self.value}]"

    @classmethod
    def get_name(cls) -> str:
        return "constant_int"

    def __eq__(self, other):
        return super().__eq__(other) and self.value == other.value

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return []

    def infer(self, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Any:
        return self.value

    def mock_exec(self, kwargs: Dict[str, Any], config: MockExecConfig) -> Any:
        return int(self.value)

    def build_ir(self, ir_id: int, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Tuple[Value, IRStatement]:
        return IntegerValue(self.value, ir_id), IRStatement(ir_id, self, [], dbg)
