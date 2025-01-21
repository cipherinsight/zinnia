from typing import List, Dict, Optional, Any, Tuple

from zenopy.builder.value import Value, FloatValue

from zenopy.compile.ir_stmt import IRStatement
from zenopy.config.mock_exec_config import MockExecConfig
from zenopy.opdef.ir_op.abstract_ir import AbstractIR
from zenopy.opdef.nocls.abstract_op import AbstractOp
from zenopy.debug.dbg_info import DebugInfo


class DivFIR(AbstractIR):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "div_f"

    @classmethod
    def get_name(cls) -> str:
        return "div_f"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("lhs"),
            AbstractOp._ParamEntry("rhs"),
        ]

    def infer(self, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Any:
        lhs, rhs = kwargs["lhs"], kwargs["rhs"]
        assert isinstance(lhs, FloatValue) and isinstance(rhs, FloatValue)
        return lhs.val() / rhs.val() if lhs.val() is not None and rhs.val() is not None else None

    def mock_exec(self, kwargs: Dict[str, Any], config: MockExecConfig) -> Any:
        return float(kwargs["lhs"]) / float(kwargs["rhs"])

    def build_ir(self, ir_id: int, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Tuple[Value, IRStatement]:
        lhs, rhs = kwargs["lhs"], kwargs["rhs"]
        assert isinstance(lhs, FloatValue) and isinstance(rhs, FloatValue)
        return FloatValue(self.infer(kwargs, dbg), ir_id), IRStatement(ir_id, self, [lhs.ptr(), rhs.ptr()], dbg)
