from typing import List, Any

from zenopy.compile.ast import ASTExpression
from zenopy.compile.ast.ast_stmt import ASTStatement
from zenopy.debug.dbg_info import DebugInfo


class ASTWhileStatement(ASTStatement):
    def __init__(
            self,
            dbg: DebugInfo,
            test_expr: ASTExpression,
            block: List[ASTStatement],
            orelse: List[ASTStatement]
    ):
        super().__init__(dbg)
        self.test_expr = test_expr
        self.block = block
        self.orelse = orelse

    def export(self) -> Any:
        return {
            "__class__": self.__class__.__name__,
            "test_expr": self.test_expr.export(),
            "block": [stmt.export() for stmt in self.block],
            "orelse": [stmt.export() for stmt in self.orelse],
        }
