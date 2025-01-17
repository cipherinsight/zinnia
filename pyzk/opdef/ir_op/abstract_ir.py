from typing import Dict, Optional, Tuple, Any

from pyzk.builder.abstract_ir_builder import AbsIRBuilderInterface
from pyzk.builder.value import Value
from pyzk.debug.dbg_info import DebugInfo
from pyzk.ir.ir_stmt import IRStatement
from pyzk.opdef.nocls.abstract_op import AbstractOp


class AbstractIR(AbstractOp):
    def __init__(self):
        super().__init__()

    def is_fixed_ir(self) -> bool:
        return False

    def build_ir(self, ir_id: int, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Tuple[Value, IRStatement]:
        raise NotImplementedError()

    def build(self, reducer: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        raise NotImplementedError("Internal Error: Unexpected call to `build` method in IR")

    def infer(self, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Any:
        raise NotImplementedError()
