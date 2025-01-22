from typing import List, Dict, Optional, Tuple

from zenopy.builder.value import Value, NoneValue, IntegerValue
from zenopy.compile.ir_stmt import IRStatement
from zenopy.opdef.ir_op.abstract_ir import AbstractIR
from zenopy.opdef.nocls.abstract_op import AbstractOp
from zenopy.debug.dbg_info import DebugInfo


class InvokeExternalIR(AbstractIR):
    def __init__(self, store_idx: int):
        super().__init__()
        self.store_idx = store_idx

    def get_signature(self) -> str:
        return f"invoke_external"

    @classmethod
    def get_name(cls) -> str:
        return "invoke_external"

    def __eq__(self, other):
        return super().__eq__(other) and self.store_idx == other.store_idx

    def is_fixed_ir(self) -> bool:
        return True

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return []

    def build_ir(self, ir_id: int, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Tuple[Value, IRStatement]:
        return NoneValue(), IRStatement(ir_id, self, [], dbg)
