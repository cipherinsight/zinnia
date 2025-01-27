from typing import List, Any

from zinnia.compile.ast.ast_chip_input import ASTChipInput
from zinnia.compile.ast.ast_comp import ASTComponent
from zinnia.compile.ast.ast_stmt import ASTStatement
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.type_sys import DTDescriptor


class ASTChip(ASTComponent):
    def __init__(
        self,
        dbg: DebugInfo,
        block: List[ASTStatement],
        inputs: List[ASTChipInput],
        return_dt: DTDescriptor,
    ):
        super().__init__(dbg)
        self.block = block
        self.inputs = inputs
        self.return_dt = return_dt

    def export(self) -> Any:
        return {
            "__class__": self.__class__.__name__,
            "block": [stmt.export() for stmt in self.block],
            "inputs": [inp.export() for inp in self.inputs],
            "return_dt": self.return_dt.export(),
        }
