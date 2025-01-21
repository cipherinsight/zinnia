from typing import List, Dict, Optional, Any, Tuple

from zenopy.builder.value import Value, IntegerValue, FloatValue

from zenopy.compile.ir_stmt import IRStatement
from zenopy.config.mock_exec_config import MockExecConfig
from zenopy.opdef.ir_op.abstract_ir import AbstractIR
from zenopy.opdef.nocls.abstract_op import AbstractOp
from zenopy.debug.dbg_info import DebugInfo


class IntCastIR(AbstractIR):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "int_cast"

    @classmethod
    def get_name(cls) -> str:
        return "int_cast"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("x")
        ]

    def infer(self, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Any:
        x = kwargs["x"]
        assert isinstance(x, FloatValue)
        return int(x.val()) if x.val() is not None else None

    def mock_exec(self, kwargs: Dict[str, Any], config: MockExecConfig) -> Any:
        return int(kwargs["x"])

    def build_ir(self, ir_id: int, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Tuple[Value, IRStatement]:
        x = kwargs["x"]
        assert isinstance(x, FloatValue)
        return IntegerValue(self.infer(kwargs, dbg), ir_id), IRStatement(ir_id, self, [x.ptr()], dbg)
