from typing import List, Dict, Optional, Any, Tuple

from pyzk.builder.value import Value, IntegerValue

from pyzk.ir.ir_stmt import IRStatement
from pyzk.opdef.ir_op.abstract_ir import AbstractIR
from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.debug.dbg_info import DebugInfo


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

    def build_ir(self, ir_id: int, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Tuple[Value, IRStatement]:
        x = kwargs["x"]
        assert isinstance(x, IntegerValue)
        return IntegerValue(self.infer(kwargs, dbg), ir_id), IRStatement(ir_id, self, [x.ptr()], dbg)
