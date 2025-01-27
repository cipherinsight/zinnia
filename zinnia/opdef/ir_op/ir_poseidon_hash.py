from typing import List, Dict, Optional, Any, Tuple

from zinnia.compile.builder.value import Value, IntegerValue
from zinnia.compile.ir.ir_stmt import IRStatement
from zinnia.config.mock_exec_config import MockExecConfig
from zinnia.opdef.ir_op.abstract_ir import AbstractIR
from zinnia.debug.dbg_info import DebugInfo


class PoseidonHashIR(AbstractIR):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "poseidon_hash"

    @classmethod
    def get_name(cls) -> str:
        return "poseidon_hash"

    def argparse(self, dbg_i: Optional[DebugInfo], args: List[Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        if len(kwargs.items()) != 0:
            raise ValueError(f"IR Operator `{self.get_name()}` does not accept keyword arguments")
        if len(args) == 0:
            raise ValueError(f"IR Operator `{self.get_name()}` requires at least one argument")
        return {f"x_{i}": args[i] for i in range(len(args))}

    def infer(self, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Any:
        return None

    def mock_exec(self, kwargs: Dict[str, Any], config: MockExecConfig) -> Any:
        # TODO: implement mock execution for poseidon hash
        return 0

    def build_ir(self, ir_id: int, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Tuple[Value, IRStatement]:
        assert all([isinstance(kwargs[f"x_{i}"], IntegerValue) for i in range(len(kwargs.items()))])
        args = [kwargs[f"x_{i}"].ptr() for i in range(len(kwargs.items()))]
        return IntegerValue(self.infer(kwargs, dbg), ir_id), IRStatement(ir_id, self, args, dbg)

    def export(self) -> Dict:
        return {}

    @staticmethod
    def import_from(data: Dict) -> 'PoseidonHashIR':
        return PoseidonHashIR()
