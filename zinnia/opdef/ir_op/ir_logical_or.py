from typing import List, Dict, Optional, Any, Tuple

from zinnia.compile.builder.value import Value, IntegerValue

from zinnia.compile.ir.ir_stmt import IRStatement
from zinnia.config.mock_exec_config import MockExecConfig
from zinnia.opdef.ir_op.abstract_ir import AbstractIR
from zinnia.opdef.nocls.abstract_op import AbstractOp
from zinnia.debug.dbg_info import DebugInfo


class LogicalOrIR(AbstractIR):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "logical_or"

    @classmethod
    def get_name(cls) -> str:
        return "logical_or"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("lhs"),
            AbstractOp._ParamEntry("rhs"),
        ]

    def infer(self, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Any:
        lhs, rhs = kwargs["lhs"], kwargs["rhs"]
        assert isinstance(lhs, IntegerValue) and isinstance(rhs, IntegerValue)
        if lhs.val() is not None and rhs.val() is not None:
            return 1 if lhs.val() != 0 or rhs.val() != 0 else 0
        elif lhs.val() is None and rhs.val() is None:
            return None
        elif lhs.val() is None and rhs.val() is not None:
            return None if rhs.val() == 0 else 1
        elif lhs.val() is not None and rhs.val() is None:
            return None if lhs.val() == 0 else 1
        raise NotImplementedError()

    def mock_exec(self, kwargs: Dict[str, Any], config: MockExecConfig) -> Any:
        return 1 if kwargs["lhs"] != 0 or kwargs["rhs"] != 0 else 0

    def build_ir(self, ir_id: int, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Tuple[Value, IRStatement]:
        lhs, rhs = kwargs["lhs"], kwargs["rhs"]
        assert isinstance(lhs, IntegerValue) and isinstance(rhs, IntegerValue)
        return IntegerValue(self.infer(kwargs, dbg), ir_id), IRStatement(ir_id, self, [lhs.ptr(), rhs.ptr()], dbg)

    def export(self) -> Dict:
        return {}

    @staticmethod
    def import_from(data: Dict) -> 'LogicalOrIR':
        return LogicalOrIR()
