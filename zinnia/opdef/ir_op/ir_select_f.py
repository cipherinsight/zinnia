from typing import List, Dict, Optional, Any, Tuple

from zinnia.compile.builder.value import Value, IntegerValue, FloatValue
from zinnia.compile.ir.ir_stmt import IRStatement
from zinnia.config.mock_exec_config import MockExecConfig
from zinnia.opdef.ir_op.abstract_ir import AbstractIR
from zinnia.opdef.nocls.abstract_op import AbstractOp
from zinnia.debug.dbg_info import DebugInfo


class SelectFIR(AbstractIR):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "select_f"

    @classmethod
    def get_name(cls) -> str:
        return "select_f"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("cond"),
            AbstractOp._ParamEntry("tv"),
            AbstractOp._ParamEntry("fv"),
        ]

    def infer(self, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Any:
        cond, tv, fv = kwargs["cond"], kwargs["tv"], kwargs["fv"]
        assert isinstance(cond, IntegerValue) and isinstance(tv, FloatValue) and isinstance(fv, FloatValue)
        if cond.val() is None:
            return None
        elif cond.val() != 0:
            return tv.val()
        else:
            return fv.val()

    def mock_exec(self, kwargs: Dict[str, Any], config: MockExecConfig) -> Any:
        return float(kwargs["tv"] if kwargs["cond"] != 0 else kwargs["fv"])

    def build_ir(self, ir_id: int, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Tuple[Value, IRStatement]:
        cond, tv, fv = kwargs["cond"], kwargs["tv"], kwargs["fv"]
        assert isinstance(cond, IntegerValue) and isinstance(tv, FloatValue) and isinstance(fv, FloatValue)
        return FloatValue(self.infer(kwargs, dbg), ir_id), IRStatement(ir_id, self, [cond.ptr(), tv.ptr(), fv.ptr()], dbg)

    def export(self) -> Dict:
        return {}

    @staticmethod
    def import_from(data: Dict) -> 'SelectFIR':
        return SelectFIR()
