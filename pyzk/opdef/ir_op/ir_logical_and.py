from typing import List, Dict, Optional, Any, Tuple

from pyzk.builder.value import Value, IntegerValue

from pyzk.ir.ir_stmt import IRStatement
from pyzk.opdef.ir_op.abstract_ir import AbstractIR
from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.debug.dbg_info import DebugInfo


class LogicalAndIR(AbstractIR):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "logical_and"

    @classmethod
    def get_name(cls) -> str:
        return "logical_and"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("lhs"),
            AbstractOp._ParamEntry("rhs"),
        ]

    def infer(self, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Any:
        lhs, rhs = kwargs["lhs"], kwargs["rhs"]
        assert isinstance(lhs, IntegerValue) and isinstance(rhs, IntegerValue)
        if lhs.val() is not None and rhs.val() is not None:
            return 1 if lhs.val() != 0 and rhs.val() != 0 else 0
        elif lhs.val() is None and rhs.val() is None:
            return None
        elif lhs.val() is None and rhs.val() is not None:
            return None if rhs.val() != 0 else 0
        elif lhs.val() is not None and rhs.val() is None:
            return None if lhs.val() != 0 else 0
        raise NotImplementedError()

    def build_ir(self, ir_id: int, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Tuple[Value, IRStatement]:
        lhs, rhs = kwargs["lhs"], kwargs["rhs"]
        assert isinstance(lhs, IntegerValue) and isinstance(rhs, IntegerValue)
        return IntegerValue(self.infer(kwargs, dbg), ir_id), IRStatement(ir_id, self, [lhs.ptr(), rhs.ptr()], dbg)
