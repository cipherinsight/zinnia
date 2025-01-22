from typing import List, Dict, Optional, Tuple

from zenopy.builder.value import Value, NoneValue, IntegerValue
from zenopy.compile.ir_stmt import IRStatement
from zenopy.opdef.ir_op.abstract_ir import AbstractIR
from zenopy.opdef.nocls.abstract_op import AbstractOp
from zenopy.debug.dbg_info import DebugInfo


class ExportExternalIIR(AbstractIR):
    def __init__(self, for_which: int, key: int | str, indices: Tuple[int, ...]):
        super().__init__()
        self.for_which = for_which
        self.indices = indices
        self.key = key

    def get_signature(self) -> str:
        return f"export_external_i[{self.for_which}][{self.key}][{', '.join(map(str, self.indices))}]"

    @classmethod
    def get_name(cls) -> str:
        return "export_external_i"

    def __eq__(self, other):
        return super().__eq__(other) and self.indices == other.indices and self.key == other.key and self.for_which == other.for_which

    def is_fixed_ir(self) -> bool:
        return True

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry('x')
        ]

    def build_ir(self, ir_id: int, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Tuple[Value, IRStatement]:
        assert isinstance(kwargs['x'], IntegerValue)
        return NoneValue(), IRStatement(ir_id, self, [kwargs['x'].ptr()], dbg)
