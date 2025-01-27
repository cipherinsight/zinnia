from typing import List, Dict, Optional, Tuple

from zinnia.compile.builder.value import Value, NoneValue
from zinnia.compile.ir.ir_stmt import IRStatement
from zinnia.compile.type_sys import DTDescriptor
from zinnia.opdef.ir_op.abstract_ir import AbstractIR
from zinnia.opdef.nocls.abstract_op import AbstractOp
from zinnia.debug.dbg_info import DebugInfo


class InvokeExternalIR(AbstractIR):
    def __init__(
            self,
            store_idx: int,
            func_name: str,
            args: List[DTDescriptor],
            kwargs: Dict[str, DTDescriptor],
    ):
        super().__init__()
        self.store_idx = store_idx
        self.func_name = func_name
        self.args = args
        self.kwargs = kwargs

    def get_signature(self) -> str:
        return f"invoke_external"

    @classmethod
    def get_name(cls) -> str:
        return "invoke_external"

    def __eq__(self, other):
        return super().__eq__(other) and self.store_idx == other.store_idx and self.func_name == other.func_name and self.args == other.args and self.kwargs == other.kwargs

    def is_fixed_ir(self) -> bool:
        return True

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return []

    def build_ir(self, ir_id: int, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Tuple[Value, IRStatement]:
        return NoneValue(), IRStatement(ir_id, self, [], dbg)

    def export(self) -> Dict:
        from zinnia.compile.type_sys.dt_factory import DTDescriptorFactory

        return {
            "store_idx": self.store_idx,
            "func_name": self.func_name,
            "args": [DTDescriptorFactory.export(arg) for arg in self.args],
            "kwargs": {k: DTDescriptorFactory.export(v) for k, v in self.kwargs.items()},
        }

    @staticmethod
    def import_from(data: Dict) -> 'InvokeExternalIR':
        from zinnia.compile.type_sys.dt_factory import DTDescriptorFactory

        args = [DTDescriptorFactory.import_from(arg) for arg in data["args"]]
        kwargs = {k: DTDescriptorFactory.import_from(v) for k, v in data["kwargs"].items()}
        return InvokeExternalIR(data["store_idx"], data["func_name"], args, kwargs)
