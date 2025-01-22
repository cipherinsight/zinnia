from typing import Any

from zenopy.compile.ast.ast_expr import ASTExpression
from zenopy.compile.ast.ast_stmt import ASTStatement
from zenopy.debug.dbg_info import DebugInfo


class ASTReturnStatement(ASTStatement):
    def __init__(self, dbg: DebugInfo, expr: ASTExpression | None):
        super().__init__(dbg)
        self.expr = expr

    def export(self) -> Any:
        return {
            "__class__": self.__class__.__name__,
            "expr": self.expr.export() if self.expr is not None else None,
        }
