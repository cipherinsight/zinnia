from typing import List, Dict, Optional, Any, Tuple

from zenopy.builder.value import Value, IntegerValue

from zenopy.compile.ir_stmt import IRStatement
from zenopy.config.mock_exec_config import MockExecConfig
from zenopy.opdef.ir_op.abstract_ir import AbstractIR
from zenopy.opdef.nocls.abstract_op import AbstractOp
from zenopy.debug.dbg_info import DebugInfo


class LogicalNotIR(AbstractIR):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "logical_not"

    @classmethod
    def get_name(cls) -> str:
        return "logical_not"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("x")
        ]

    def infer(self, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Any:
        x = kwargs["x"]
        assert isinstance(x, IntegerValue)
        return (1 if x.val() == 0 else 0) if x.val() is not None else None

    def mock_exec(self, kwargs: Dict[str, Any], config: MockExecConfig) -> Any:
        return 1 if kwargs["x"] == 0 else 0

    def build_ir(self, ir_id: int, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Tuple[Value, IRStatement]:
        x = kwargs["x"]
        assert isinstance(x, IntegerValue)
        return IntegerValue(self.infer(kwargs, dbg), ir_id), IRStatement(ir_id, self, [x.ptr()], dbg)
