from typing import Dict, Optional, Tuple, Any

from zinnia.compile.builder.abstract_ir_builder import AbsIRBuilderInterface
from zinnia.compile.builder.value import Value
from zinnia.config.mock_exec_config import MockExecConfig
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.ir.ir_stmt import IRStatement
from zinnia.opdef.abstract.abstract_op import AbstractOp


class AbstractIR(AbstractOp):
    def __init__(self):
        super().__init__()

    def is_fixed_ir(self) -> bool:
        return False

    def build_ir(self, ir_id: int, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Tuple[Value, IRStatement]:
        raise NotImplementedError()

    def build(self, builder: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        raise NotImplementedError("Internal Error: Unexpected call to `build` method in IR")

    def infer(self, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Any:
        raise NotImplementedError()

    def mock_exec(self, kwargs: Dict[str, Any], config: MockExecConfig) -> Any:
        raise NotImplementedError()

    def export(self) -> Dict:
        raise NotImplementedError()

    @staticmethod
    def import_from(data: Dict) -> 'AbstractIR':
        raise NotImplementedError()
