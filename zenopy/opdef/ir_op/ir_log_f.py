from typing import List, Dict, Optional, Any, Tuple

import math

from zenopy.builder.value import Value, FloatValue
from zenopy.debug.exception import StaticInferenceError

from zenopy.ir.ir_stmt import IRStatement
from zenopy.opdef.ir_op.abstract_ir import AbstractIR
from zenopy.opdef.nocls.abstract_op import AbstractOp
from zenopy.debug.dbg_info import DebugInfo


class LogFIR(AbstractIR):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "log_f"

    @classmethod
    def get_name(cls) -> str:
        return "log_f"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("x")
        ]

    def infer(self, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Any:
        x = kwargs["x"]
        assert isinstance(x, FloatValue)
        if x.val() is not None and x.val() <= 0:
            raise StaticInferenceError(dbg, f"Invalid argument for `log`: argument ({x.val()}) inferred to be negative or zero")
        return math.log(x.val()) if x.val() is not None else None

    def build_ir(self, ir_id: int, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Tuple[Value, IRStatement]:
        x = kwargs["x"]
        assert isinstance(x, FloatValue)
        return FloatValue(self.infer(kwargs, dbg), ir_id), IRStatement(ir_id, self, [x.ptr()], dbg)
