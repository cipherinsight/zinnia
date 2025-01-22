from typing import List, Any

from zenopy.compile.ast.ast_anno import ASTAnnotation
from zenopy.compile.ast.ast_chip_input import ASTChipInput
from zenopy.compile.ast.ast_comp import ASTComponent
from zenopy.compile.ast.ast_stmt import ASTStatement
from zenopy.debug.dbg_info import DebugInfo


class ASTChip(ASTComponent):
    def __init__(
        self,
        dbg: DebugInfo,
        block: List[ASTStatement],
        inputs: List[ASTChipInput],
        return_anno: ASTAnnotation,
    ):
        super().__init__(dbg)
        self.block = block
        self.inputs = inputs
        self.return_anno = return_anno

    def export(self) -> Any:
        return {
            "__class__": self.__class__.__name__,
            "block": [stmt.export() for stmt in self.block],
            "inputs": [inp.export() for inp in self.inputs],
            "return_anno": self.return_anno.export(),
        }
