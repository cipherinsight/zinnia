from typing import List, Dict, Optional, Any, Tuple

from zenopy.builder.value import Value, IntegerValue

from zenopy.compile.ir_stmt import IRStatement
from zenopy.opdef.ir_op.abstract_ir import AbstractIR
from zenopy.debug.dbg_info import DebugInfo


class HashIR(AbstractIR):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "hash"

    @classmethod
    def get_name(cls) -> str:
        return "hash"

    def argparse(self, dbg_i: Optional[DebugInfo], args: List[Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        if len(kwargs.items()) != 0:
            raise ValueError(f"IR Operator `{self.get_name()}` does not accept keyword arguments")
        if len(args) == 0:
            raise ValueError(f"IR Operator `{self.get_name()}` requires at least one argument")
        return {f"x_{i}": args[i] for i in range(len(args))}

    def infer(self, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Any:
        return None

    def build_ir(self, ir_id: int, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Tuple[Value, IRStatement]:
        assert all([isinstance(kwargs[f"x_{i}"], IntegerValue) for i in range(len(kwargs.items()))])
        args = [kwargs[f"x_{i}"].ptr() for i in range(len(kwargs.items()))]
        return IntegerValue(self.infer(kwargs, dbg), ir_id), IRStatement(ir_id, self, args, dbg)
