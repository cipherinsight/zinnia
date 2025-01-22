from typing import List, Any

from zenopy.compile.ast.ast_comp import ASTComponent
from zenopy.compile.ast.ast_circuit_input import ASTCircuitInput
from zenopy.compile.ast.ast_stmt import ASTStatement
from zenopy.debug.dbg_info import DebugInfo


class ASTCircuit(ASTComponent):
    def __init__(
        self,
        dbg: DebugInfo,
        block: List[ASTStatement],
        inputs: List[ASTCircuitInput]
    ):
        super().__init__(dbg)
        self.block = block
        self.inputs = inputs

    def export(self) -> Any:
        return {
            "__class__": self.__class__.__name__,
            "block": [stmt.export() for stmt in self.block],
            "inputs": [inp.export() for inp in self.inputs],
        }
