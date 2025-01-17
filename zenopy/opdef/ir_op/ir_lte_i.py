from typing import List, Dict, Optional, Any, Tuple

from zenopy.builder.value import Value, IntegerValue

from zenopy.ir.ir_stmt import IRStatement
from zenopy.opdef.ir_op.abstract_ir import AbstractIR
from zenopy.opdef.nocls.abstract_op import AbstractOp
from zenopy.debug.dbg_info import DebugInfo


class LessThanOrEqualIIR(AbstractIR):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "lte_i"

    @classmethod
    def get_name(cls) -> str:
        return "lte_i"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("lhs"),
            AbstractOp._ParamEntry("rhs"),
        ]

    def infer(self, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Any:
        lhs, rhs = kwargs["lhs"], kwargs["rhs"]
        assert isinstance(lhs, IntegerValue) and isinstance(rhs, IntegerValue)
        return (1 if lhs.val() <= rhs.val() else 0) if lhs.val() is not None and rhs.val() is not None else None

    def build_ir(self, ir_id: int, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Tuple[Value, IRStatement]:
        lhs, rhs = kwargs["lhs"], kwargs["rhs"]
        assert isinstance(lhs, IntegerValue) and isinstance(rhs, IntegerValue)
        return IntegerValue(self.infer(kwargs, dbg), ir_id), IRStatement(ir_id, self, [lhs.ptr(), rhs.ptr()], dbg)
