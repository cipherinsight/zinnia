from typing import List, Dict, Optional, Any, Tuple

from zenopy.builder.value import Value, IntegerValue, NoneValue

from zenopy.ir.ir_stmt import IRStatement
from zenopy.opdef.ir_op.abstract_ir import AbstractIR
from zenopy.opdef.nocls.abstract_op import AbstractOp
from zenopy.debug.dbg_info import DebugInfo


class AssertIR(AbstractIR):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "assert"

    @classmethod
    def get_name(cls) -> str:
        return "assert"

    def is_fixed_ir(self) -> bool:
        return True

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("test")
        ]

    def infer(self, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Any:
        test = kwargs["test"]
        assert isinstance(test, IntegerValue)
        return None

    def build_ir(self, ir_id: int, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Tuple[Value, IRStatement]:
        test = kwargs["test"]
        assert isinstance(test, IntegerValue)
        return NoneValue(), IRStatement(ir_id, self, [test.ptr()], dbg)
