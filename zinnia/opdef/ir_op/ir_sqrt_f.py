from typing import List, Dict, Optional, Any, Tuple

import math

from zinnia.compile.builder.value import Value, FloatValue
from zinnia.config.mock_exec_config import MockExecConfig
from zinnia.debug.exception import StaticInferenceError

from zinnia.compile.ir.ir_stmt import IRStatement
from zinnia.opdef.ir_op.abstract_ir import AbstractIR
from zinnia.opdef.nocls.abstract_op import AbstractOp
from zinnia.debug.dbg_info import DebugInfo


class SqrtFIR(AbstractIR):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "sqrt_f"

    @classmethod
    def get_name(cls) -> str:
        return "sqrt_f"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("x")
        ]

    def infer(self, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Any:
        x = kwargs["x"]
        assert isinstance(x, FloatValue)
        if x.val() is not None and x.val() < 0:
            raise StaticInferenceError(dbg, f"A negative value is inferred at compiler time on `sqrt`. Cannot take square root of negative number: {x.val()}")
        return math.sqrt(x.val()) if x.val() is not None else None

    def mock_exec(self, kwargs: Dict[str, Any], config: MockExecConfig) -> Any:
        return float(math.sqrt(kwargs["x"]))

    def build_ir(self, ir_id: int, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Tuple[Value, IRStatement]:
        x = kwargs["x"]
        assert isinstance(x, FloatValue)
        return FloatValue(self.infer(kwargs, dbg), ir_id), IRStatement(ir_id, self, [x.ptr()], dbg)

    def export(self) -> Dict:
        return {}

    @staticmethod
    def import_from(data: Dict) -> 'SqrtFIR':
        return SqrtFIR()
